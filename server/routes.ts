import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import sharp from "sharp";
import path from "path";
import fs from "fs";
import { insertPhotoSchema, insertAdventureSchema, insertMemoryBookSchema, insertMemoryBookPhotoSchema } from "@shared/schema";

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(process.cwd(), "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer for file uploads
const upload = multer({
  dest: uploadsDir,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  },
});

export async function registerRoutes(app: Express): Promise<Server> {
  
  // Serve uploaded images
  app.use('/uploads', (req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    next();
  });
  app.use('/uploads', express.static(uploadsDir));

  // Stats endpoint
  app.get("/api/stats", async (req, res) => {
    try {
      const stats = await storage.getStats();
      res.json(stats);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch stats" });
    }
  });

  // Photos endpoints
  app.get("/api/photos", async (req, res) => {
    try {
      const { category, adventure } = req.query;
      let photos;
      
      if (category) {
        photos = await storage.getPhotosByCategory(category as string);
      } else if (adventure) {
        photos = await storage.getPhotosByAdventure(parseInt(adventure as string));
      } else {
        photos = await storage.getPhotos();
      }
      
      res.json(photos);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch photos" });
    }
  });

  app.get("/api/photos/:id", async (req, res) => {
    try {
      const photo = await storage.getPhoto(parseInt(req.params.id));
      if (!photo) {
        return res.status(404).json({ message: "Photo not found" });
      }
      res.json(photo);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch photo" });
    }
  });

  app.post("/api/photos", upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ message: "No image file provided" });
      }

      // Process image with Sharp
      const filename = `${Date.now()}-${Math.random().toString(36).substring(7)}.jpg`;
      const outputPath = path.join(uploadsDir, filename);
      
      await sharp(req.file.path)
        .resize(1200, 1200, { 
          fit: 'inside', 
          withoutEnlargement: true 
        })
        .jpeg({ quality: 85 })
        .toFile(outputPath);

      // Clean up original file
      fs.unlinkSync(req.file.path);

      // Parse and validate request body
      const photoData = {
        filename,
        originalName: req.file.originalname,
        url: `/uploads/${filename}`,
        title: req.body.title || req.file.originalname,
        location: req.body.location || null,
        category: req.body.category || 'other',
        adventureId: req.body.adventureId ? parseInt(req.body.adventureId) : null,
        isLiked: false,
      };

      const validatedData = insertPhotoSchema.parse(photoData);
      const photo = await storage.createPhoto(validatedData);
      
      res.status(201).json(photo);
    } catch (error) {
      console.error('Photo upload error:', error);
      if (req.file && fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      res.status(500).json({ message: "Failed to upload photo" });
    }
  });

  app.patch("/api/photos/:id", async (req, res) => {
    try {
      const photoId = parseInt(req.params.id);
      const updates = req.body;
      
      const photo = await storage.updatePhoto(photoId, updates);
      if (!photo) {
        return res.status(404).json({ message: "Photo not found" });
      }
      
      res.json(photo);
    } catch (error) {
      res.status(500).json({ message: "Failed to update photo" });
    }
  });

  app.delete("/api/photos/:id", async (req, res) => {
    try {
      const photoId = parseInt(req.params.id);
      const photo = await storage.getPhoto(photoId);
      
      if (!photo) {
        return res.status(404).json({ message: "Photo not found" });
      }

      // Delete file from disk
      const filePath = path.join(uploadsDir, photo.filename);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }

      const deleted = await storage.deletePhoto(photoId);
      if (!deleted) {
        return res.status(404).json({ message: "Photo not found" });
      }
      
      res.json({ message: "Photo deleted successfully" });
    } catch (error) {
      res.status(500).json({ message: "Failed to delete photo" });
    }
  });

  // Adventures endpoints
  app.get("/api/adventures", async (req, res) => {
    try {
      const adventures = await storage.getAdventures();
      
      // Enrich with photo counts
      const enrichedAdventures = await Promise.all(
        adventures.map(async (adventure) => {
          const photos = await storage.getPhotosByAdventure(adventure.id);
          return {
            ...adventure,
            photoCount: photos.length,
            coverPhoto: adventure.coverPhotoId ? await storage.getPhoto(adventure.coverPhotoId) : photos[0] || null,
          };
        })
      );
      
      res.json(enrichedAdventures);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch adventures" });
    }
  });

  app.get("/api/adventures/:id", async (req, res) => {
    try {
      const adventure = await storage.getAdventure(parseInt(req.params.id));
      if (!adventure) {
        return res.status(404).json({ message: "Adventure not found" });
      }
      
      const photos = await storage.getPhotosByAdventure(adventure.id);
      res.json({ ...adventure, photos });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch adventure" });
    }
  });

  app.post("/api/adventures", async (req, res) => {
    try {
      const validatedData = insertAdventureSchema.parse(req.body);
      const adventure = await storage.createAdventure(validatedData);
      res.status(201).json(adventure);
    } catch (error) {
      res.status(400).json({ message: "Invalid adventure data" });
    }
  });

  // Memory Books endpoints
  app.get("/api/memory-books", async (req, res) => {
    try {
      const memoryBooks = await storage.getMemoryBooks();
      
      // Enrich with photo counts and cover photos
      const enrichedMemoryBooks = await Promise.all(
        memoryBooks.map(async (memoryBook) => {
          const memoryBookPhotos = await storage.getMemoryBookPhotos(memoryBook.id);
          const coverPhoto = memoryBook.coverPhotoId 
            ? await storage.getPhoto(memoryBook.coverPhotoId)
            : memoryBookPhotos.length > 0 
              ? await storage.getPhoto(memoryBookPhotos[0].photoId)
              : null;
          
          return {
            ...memoryBook,
            photoCount: memoryBookPhotos.length,
            coverPhoto,
          };
        })
      );
      
      res.json(enrichedMemoryBooks);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch memory books" });
    }
  });

  app.get("/api/memory-books/:id", async (req, res) => {
    try {
      const memoryBook = await storage.getMemoryBook(parseInt(req.params.id));
      if (!memoryBook) {
        return res.status(404).json({ message: "Memory book not found" });
      }
      
      const memoryBookPhotos = await storage.getMemoryBookPhotos(memoryBook.id);
      const photos = await Promise.all(
        memoryBookPhotos.map(async (mbp) => {
          const photo = await storage.getPhoto(mbp.photoId);
          return { ...photo, order: mbp.order };
        })
      );
      
      res.json({ ...memoryBook, photos });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch memory book" });
    }
  });

  app.post("/api/memory-books", async (req, res) => {
    try {
      const validatedData = insertMemoryBookSchema.parse(req.body);
      const memoryBook = await storage.createMemoryBook(validatedData);
      res.status(201).json(memoryBook);
    } catch (error) {
      res.status(400).json({ message: "Invalid memory book data" });
    }
  });

  app.post("/api/memory-books/:id/photos", async (req, res) => {
    try {
      const memoryBookId = parseInt(req.params.id);
      const validatedData = insertMemoryBookPhotoSchema.parse({
        ...req.body,
        memoryBookId,
      });
      
      const memoryBookPhoto = await storage.addPhotoToMemoryBook(validatedData);
      res.status(201).json(memoryBookPhoto);
    } catch (error) {
      res.status(400).json({ message: "Failed to add photo to memory book" });
    }
  });

  app.delete("/api/memory-books/:id/photos/:photoId", async (req, res) => {
    try {
      const memoryBookId = parseInt(req.params.id);
      const photoId = parseInt(req.params.photoId);
      
      const removed = await storage.removePhotoFromMemoryBook(memoryBookId, photoId);
      if (!removed) {
        return res.status(404).json({ message: "Photo not found in memory book" });
      }
      
      res.json({ message: "Photo removed from memory book" });
    } catch (error) {
      res.status(500).json({ message: "Failed to remove photo from memory book" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
