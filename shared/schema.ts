import { pgTable, text, serial, integer, boolean, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const photos = pgTable("photos", {
  id: serial("id").primaryKey(),
  filename: text("filename").notNull(),
  originalName: text("original_name").notNull(),
  url: text("url").notNull(),
  title: text("title"),
  location: text("location"),
  category: text("category").notNull(),
  adventureId: integer("adventure_id"),
  isLiked: boolean("is_liked").default(false),
  uploadedAt: timestamp("uploaded_at").defaultNow(),
});

export const adventures = pgTable("adventures", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  location: text("location").notNull(),
  category: text("category").notNull(),
  date: timestamp("date").notNull(),
  description: text("description"),
  coverPhotoId: integer("cover_photo_id"),
});

export const memoryBooks = pgTable("memory_books", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description"),
  coverPhotoId: integer("cover_photo_id"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const memoryBookPhotos = pgTable("memory_book_photos", {
  id: serial("id").primaryKey(),
  memoryBookId: integer("memory_book_id").notNull(),
  photoId: integer("photo_id").notNull(),
  order: integer("order").notNull(),
});

export const insertPhotoSchema = createInsertSchema(photos).omit({
  id: true,
  uploadedAt: true,
});

export const insertAdventureSchema = createInsertSchema(adventures).omit({
  id: true,
});

export const insertMemoryBookSchema = createInsertSchema(memoryBooks).omit({
  id: true,
  createdAt: true,
});

export const insertMemoryBookPhotoSchema = createInsertSchema(memoryBookPhotos).omit({
  id: true,
});

export type Photo = typeof photos.$inferSelect;
export type InsertPhoto = z.infer<typeof insertPhotoSchema>;
export type Adventure = typeof adventures.$inferSelect;
export type InsertAdventure = z.infer<typeof insertAdventureSchema>;
export type MemoryBook = typeof memoryBooks.$inferSelect;
export type InsertMemoryBook = z.infer<typeof insertMemoryBookSchema>;
export type MemoryBookPhoto = typeof memoryBookPhotos.$inferSelect;
export type InsertMemoryBookPhoto = z.infer<typeof insertMemoryBookPhotoSchema>;

export const categories = [
  "hiking",
  "camping", 
  "climbing",
  "water-sports",
  "wildlife",
  "landscape",
  "other"
] as const;

export type Category = typeof categories[number];
