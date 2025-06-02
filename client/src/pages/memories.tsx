import { useQuery } from "@tanstack/react-query";
import { Heart, Plus, Book, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import MemoryBookModal from "@/components/memory-book-modal";
import type { MemoryBook, Photo } from "@shared/schema";

interface MemoryBookWithPhotos extends MemoryBook {
  photoCount: number;
  coverPhoto?: Photo | null;
}

export default function Memories() {
  const [isMemoryModalOpen, setIsMemoryModalOpen] = useState(false);

  const { data: memoryBooks = [] } = useQuery<MemoryBookWithPhotos[]>({
    queryKey: ['/api/memory-books'],
  });

  return (
    <>
      {/* Header */}
      <header className="bg-forest text-white p-4 flex items-center justify-between sticky top-0 z-50 shadow-lg">
        <div className="flex items-center space-x-3">
          <Heart className="text-2xl text-sage" />
          <h1 className="text-xl font-bold">Memories</h1>
        </div>
        <Button 
          onClick={() => setIsMemoryModalOpen(true)}
          size="sm"
          className="bg-sage hover:bg-sage/90 text-forest"
        >
          <Plus className="h-4 w-4 mr-1" />
          Create
        </Button>
      </header>

      <main className="pb-20">
        {/* Quick Actions */}
        <section className="p-4">
          <div className="grid grid-cols-2 gap-3">
            <Button 
              onClick={() => setIsMemoryModalOpen(true)}
              className="bg-gradient-to-r from-sunset to-sunset/80 hover:from-sunset/90 hover:to-sunset/70 text-white p-4 h-auto rounded-xl shadow-lg flex items-center justify-center space-x-2 hover:shadow-xl transition-all transform hover:scale-105"
            >
              <Book className="h-5 w-5" />
              <span className="font-semibold">New Memory Book</span>
            </Button>
            <Button 
              variant="outline"
              className="border-forest text-forest hover:bg-forest hover:text-white p-4 h-auto rounded-xl shadow-lg flex items-center justify-center space-x-2 hover:shadow-xl transition-all transform hover:scale-105"
            >
              <Calendar className="h-5 w-5" />
              <span className="font-semibold">By Date</span>
            </Button>
          </div>
        </section>

        {/* Memory Books Grid */}
        {memoryBooks.length > 0 ? (
          <section className="px-4 mb-6">
            <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200 mb-4">Your Memory Books</h2>
            
            <div className="grid grid-cols-2 gap-4">
              {memoryBooks.map((memoryBook) => (
                <div 
                  key={memoryBook.id} 
                  className="bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
                >
                  <div className="aspect-square bg-gray-200 dark:bg-gray-700 relative overflow-hidden">
                    {memoryBook.coverPhoto ? (
                      <img 
                        src={memoryBook.coverPhoto.url} 
                        alt={memoryBook.title}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Heart className="h-12 w-12 text-gray-400" />
                      </div>
                    )}
                    <div className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded-full">
                      {memoryBook.photoCount}
                    </div>
                  </div>
                  
                  <div className="p-3">
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-1 truncate">
                      {memoryBook.title}
                    </h3>
                    {memoryBook.description && (
                      <p className="text-xs text-stone mb-2 line-clamp-2">
                        {memoryBook.description}
                      </p>
                    )}
                    <div className="flex items-center text-xs text-stone">
                      <Calendar className="h-3 w-3 mr-1" />
                      {new Date(memoryBook.createdAt!).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        ) : (
          <section className="px-4 py-8 text-center">
            <Heart className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-600 mb-2">Create Your First Memory</h3>
            <p className="text-gray-500 mb-4">
              Collect your favorite adventure photos into beautiful memory books to cherish forever.
            </p>
            <Button 
              onClick={() => setIsMemoryModalOpen(true)}
              className="bg-forest hover:bg-forest/90 text-white"
            >
              <Plus className="h-4 w-4 mr-2" />
              Create Memory Book
            </Button>
          </section>
        )}
      </main>

      {/* Memory Book Modal */}
      {isMemoryModalOpen && (
        <MemoryBookModal onClose={() => setIsMemoryModalOpen(false)} />
      )}
    </>
  );
}
