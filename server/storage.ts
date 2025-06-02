import { 
  Photo, 
  InsertPhoto, 
  Adventure, 
  InsertAdventure, 
  MemoryBook, 
  InsertMemoryBook,
  MemoryBookPhoto,
  InsertMemoryBookPhoto 
} from "@shared/schema";

export interface IStorage {
  // Photos
  getPhotos(): Promise<Photo[]>;
  getPhoto(id: number): Promise<Photo | undefined>;
  getPhotosByCategory(category: string): Promise<Photo[]>;
  getPhotosByAdventure(adventureId: number): Promise<Photo[]>;
  createPhoto(photo: InsertPhoto): Promise<Photo>;
  updatePhoto(id: number, updates: Partial<Photo>): Promise<Photo | undefined>;
  deletePhoto(id: number): Promise<boolean>;
  
  // Adventures
  getAdventures(): Promise<Adventure[]>;
  getAdventure(id: number): Promise<Adventure | undefined>;
  createAdventure(adventure: InsertAdventure): Promise<Adventure>;
  updateAdventure(id: number, updates: Partial<Adventure>): Promise<Adventure | undefined>;
  deleteAdventure(id: number): Promise<boolean>;
  
  // Memory Books
  getMemoryBooks(): Promise<MemoryBook[]>;
  getMemoryBook(id: number): Promise<MemoryBook | undefined>;
  createMemoryBook(memoryBook: InsertMemoryBook): Promise<MemoryBook>;
  updateMemoryBook(id: number, updates: Partial<MemoryBook>): Promise<MemoryBook | undefined>;
  deleteMemoryBook(id: number): Promise<boolean>;
  
  // Memory Book Photos
  getMemoryBookPhotos(memoryBookId: number): Promise<MemoryBookPhoto[]>;
  addPhotoToMemoryBook(memoryBookPhoto: InsertMemoryBookPhoto): Promise<MemoryBookPhoto>;
  removePhotoFromMemoryBook(memoryBookId: number, photoId: number): Promise<boolean>;
  
  // Stats
  getStats(): Promise<{
    totalPhotos: number;
    totalAdventures: number;
    totalMemories: number;
  }>;
}

export class MemStorage implements IStorage {
  private photos: Map<number, Photo>;
  private adventures: Map<number, Adventure>;
  private memoryBooks: Map<number, MemoryBook>;
  private memoryBookPhotos: Map<number, MemoryBookPhoto>;
  private currentPhotoId: number;
  private currentAdventureId: number;
  private currentMemoryBookId: number;
  private currentMemoryBookPhotoId: number;

  constructor() {
    this.photos = new Map();
    this.adventures = new Map();
    this.memoryBooks = new Map();
    this.memoryBookPhotos = new Map();
    this.currentPhotoId = 1;
    this.currentAdventureId = 1;
    this.currentMemoryBookId = 1;
    this.currentMemoryBookPhotoId = 1;
  }

  // Photos
  async getPhotos(): Promise<Photo[]> {
    return Array.from(this.photos.values()).sort((a, b) => 
      new Date(b.uploadedAt!).getTime() - new Date(a.uploadedAt!).getTime()
    );
  }

  async getPhoto(id: number): Promise<Photo | undefined> {
    return this.photos.get(id);
  }

  async getPhotosByCategory(category: string): Promise<Photo[]> {
    return Array.from(this.photos.values())
      .filter(photo => photo.category === category)
      .sort((a, b) => new Date(b.uploadedAt!).getTime() - new Date(a.uploadedAt!).getTime());
  }

  async getPhotosByAdventure(adventureId: number): Promise<Photo[]> {
    return Array.from(this.photos.values())
      .filter(photo => photo.adventureId === adventureId)
      .sort((a, b) => new Date(b.uploadedAt!).getTime() - new Date(a.uploadedAt!).getTime());
  }

  async createPhoto(insertPhoto: InsertPhoto): Promise<Photo> {
    const id = this.currentPhotoId++;
    const photo: Photo = {
      ...insertPhoto,
      id,
      uploadedAt: new Date(),
    };
    this.photos.set(id, photo);
    return photo;
  }

  async updatePhoto(id: number, updates: Partial<Photo>): Promise<Photo | undefined> {
    const photo = this.photos.get(id);
    if (!photo) return undefined;
    
    const updatedPhoto = { ...photo, ...updates };
    this.photos.set(id, updatedPhoto);
    return updatedPhoto;
  }

  async deletePhoto(id: number): Promise<boolean> {
    return this.photos.delete(id);
  }

  // Adventures
  async getAdventures(): Promise<Adventure[]> {
    return Array.from(this.adventures.values()).sort((a, b) => 
      new Date(b.date).getTime() - new Date(a.date).getTime()
    );
  }

  async getAdventure(id: number): Promise<Adventure | undefined> {
    return this.adventures.get(id);
  }

  async createAdventure(insertAdventure: InsertAdventure): Promise<Adventure> {
    const id = this.currentAdventureId++;
    const adventure: Adventure = { ...insertAdventure, id };
    this.adventures.set(id, adventure);
    return adventure;
  }

  async updateAdventure(id: number, updates: Partial<Adventure>): Promise<Adventure | undefined> {
    const adventure = this.adventures.get(id);
    if (!adventure) return undefined;
    
    const updatedAdventure = { ...adventure, ...updates };
    this.adventures.set(id, updatedAdventure);
    return updatedAdventure;
  }

  async deleteAdventure(id: number): Promise<boolean> {
    return this.adventures.delete(id);
  }

  // Memory Books
  async getMemoryBooks(): Promise<MemoryBook[]> {
    return Array.from(this.memoryBooks.values()).sort((a, b) => 
      new Date(b.createdAt!).getTime() - new Date(a.createdAt!).getTime()
    );
  }

  async getMemoryBook(id: number): Promise<MemoryBook | undefined> {
    return this.memoryBooks.get(id);
  }

  async createMemoryBook(insertMemoryBook: InsertMemoryBook): Promise<MemoryBook> {
    const id = this.currentMemoryBookId++;
    const memoryBook: MemoryBook = {
      ...insertMemoryBook,
      id,
      createdAt: new Date(),
    };
    this.memoryBooks.set(id, memoryBook);
    return memoryBook;
  }

  async updateMemoryBook(id: number, updates: Partial<MemoryBook>): Promise<MemoryBook | undefined> {
    const memoryBook = this.memoryBooks.get(id);
    if (!memoryBook) return undefined;
    
    const updatedMemoryBook = { ...memoryBook, ...updates };
    this.memoryBooks.set(id, updatedMemoryBook);
    return updatedMemoryBook;
  }

  async deleteMemoryBook(id: number): Promise<boolean> {
    return this.memoryBooks.delete(id);
  }

  // Memory Book Photos
  async getMemoryBookPhotos(memoryBookId: number): Promise<MemoryBookPhoto[]> {
    return Array.from(this.memoryBookPhotos.values())
      .filter(mbp => mbp.memoryBookId === memoryBookId)
      .sort((a, b) => a.order - b.order);
  }

  async addPhotoToMemoryBook(insertMemoryBookPhoto: InsertMemoryBookPhoto): Promise<MemoryBookPhoto> {
    const id = this.currentMemoryBookPhotoId++;
    const memoryBookPhoto: MemoryBookPhoto = { ...insertMemoryBookPhoto, id };
    this.memoryBookPhotos.set(id, memoryBookPhoto);
    return memoryBookPhoto;
  }

  async removePhotoFromMemoryBook(memoryBookId: number, photoId: number): Promise<boolean> {
    const entry = Array.from(this.memoryBookPhotos.entries())
      .find(([_, mbp]) => mbp.memoryBookId === memoryBookId && mbp.photoId === photoId);
    
    if (entry) {
      return this.memoryBookPhotos.delete(entry[0]);
    }
    return false;
  }

  // Stats
  async getStats(): Promise<{ totalPhotos: number; totalAdventures: number; totalMemories: number; }> {
    return {
      totalPhotos: this.photos.size,
      totalAdventures: this.adventures.size,
      totalMemories: this.memoryBooks.size,
    };
  }
}

export const storage = new MemStorage();
