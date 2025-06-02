import { useQuery } from "@tanstack/react-query";
import { Mountain, Search, UserCircle, Heart, Plus, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import PhotoGrid from "@/components/photo-grid";
import PhotoModal from "@/components/photo-modal";
import PhotoUpload from "@/components/photo-upload";
import MemoryBookModal from "@/components/memory-book-modal";
import CategoryFilter from "@/components/category-filter";
import AdventureCard from "@/components/adventure-card";
import type { Photo, Adventure, MemoryBook } from "@shared/schema";

interface StatsData {
  totalPhotos: number;
  totalAdventures: number;
  totalMemories: number;
}

interface AdventureWithPhotos extends Adventure {
  photoCount: number;
  coverPhoto?: Photo | null;
}

interface MemoryBookWithPhotos extends MemoryBook {
  photoCount: number;
  coverPhoto?: Photo | null;
}

export default function Home() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);
  const [isPhotoUploadOpen, setIsPhotoUploadOpen] = useState(false);
  const [isMemoryModalOpen, setIsMemoryModalOpen] = useState(false);

  const { data: stats } = useQuery<StatsData>({
    queryKey: ['/api/stats'],
  });

  const { data: photos = [] } = useQuery<Photo[]>({
    queryKey: selectedCategory === "all" 
      ? ['/api/photos'] 
      : ['/api/photos', { category: selectedCategory }],
  });

  const { data: adventures = [] } = useQuery<AdventureWithPhotos[]>({
    queryKey: ['/api/adventures'],
  });

  const { data: memoryBooks = [] } = useQuery<MemoryBookWithPhotos[]>({
    queryKey: ['/api/memory-books'],
  });

  const latestPhotos = photos.slice(0, 6);
  const recentAdventures = adventures.slice(0, 2);
  const recentMemoryBooks = memoryBooks.slice(0, 3);

  return (
    <>
      {/* Header */}
      <header className="bg-forest text-white p-4 flex items-center justify-between sticky top-0 z-50 shadow-lg">
        <div className="flex items-center space-x-3">
          <Mountain className="text-2xl text-sage" />
          <h1 className="text-xl font-bold">AdventureSnap</h1>
        </div>
        <div className="flex items-center space-x-4">
          <Button variant="ghost" size="icon" className="text-sage hover:text-white">
            <Search className="h-5 w-5" />
          </Button>
          <Button variant="ghost" size="icon" className="text-sage hover:text-white">
            <UserCircle className="h-6 w-6" />
          </Button>
        </div>
      </header>

      <main className="pb-20">
        {/* Stats Overview */}
        <section className="p-4 bg-gradient-to-r from-sage/20 to-sky/20">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
              <div className="text-2xl font-bold text-forest dark:text-sage">
                {stats?.totalPhotos || 0}
              </div>
              <div className="text-xs text-stone">Photos</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
              <div className="text-2xl font-bold text-sky">
                {stats?.totalAdventures || 0}
              </div>
              <div className="text-xs text-stone">Adventures</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
              <div className="text-2xl font-bold text-sunset">
                {stats?.totalMemories || 0}
              </div>
              <div className="text-xs text-stone">Memories</div>
            </div>
          </div>
        </section>

        {/* Category Filter */}
        <section className="p-4">
          <CategoryFilter 
            selectedCategory={selectedCategory}
            onCategoryChange={setSelectedCategory}
          />
        </section>

        {/* Quick Actions */}
        <section className="px-4 mb-6">
          <div className="grid grid-cols-2 gap-3">
            <Button 
              onClick={() => setIsMemoryModalOpen(true)}
              className="bg-gradient-to-r from-sunset to-sunset/80 hover:from-sunset/90 hover:to-sunset/70 text-white p-4 h-auto rounded-xl shadow-lg flex items-center justify-center space-x-2 hover:shadow-xl transition-all transform hover:scale-105"
            >
              <Heart className="h-5 w-5" />
              <span className="font-semibold">Create Memory</span>
            </Button>
            <Button 
              onClick={() => setIsPhotoUploadOpen(true)}
              className="bg-gradient-to-r from-sky to-sky/80 hover:from-sky/90 hover:to-sky/70 text-white p-4 h-auto rounded-xl shadow-lg flex items-center justify-center space-x-2 hover:shadow-xl transition-all transform hover:scale-105"
            >
              <Plus className="h-5 w-5" />
              <span className="font-semibold">Add Photos</span>
            </Button>
          </div>
        </section>

        {/* Recent Adventures */}
        {recentAdventures.length > 0 && (
          <section className="px-4 mb-6">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200">Recent Adventures</h2>
              <Button variant="link" className="text-forest p-0 h-auto">View All</Button>
            </div>
            
            <div className="space-y-3">
              {recentAdventures.map((adventure) => (
                <AdventureCard key={adventure.id} adventure={adventure} />
              ))}
            </div>
          </section>
        )}

        {/* Latest Photos */}
        {latestPhotos.length > 0 && (
          <section className="px-4 mb-6">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200">Latest Photos</h2>
              <Button variant="link" className="text-forest p-0 h-auto">View All</Button>
            </div>

            <PhotoGrid 
              photos={latestPhotos} 
              onPhotoClick={setSelectedPhoto}
            />
          </section>
        )}

        {/* Memory Books */}
        {recentMemoryBooks.length > 0 && (
          <section className="px-4 mb-6">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200">Memory Books</h2>
              <Button variant="link" className="text-forest p-0 h-auto">View All</Button>
            </div>

            <div className="flex space-x-4 overflow-x-auto pb-2">
              {recentMemoryBooks.map((memoryBook) => (
                <div key={memoryBook.id} className="flex-shrink-0 w-32 cursor-pointer">
                  <div className="w-full h-40 bg-gray-200 dark:bg-gray-700 rounded-lg shadow-md mb-2 overflow-hidden">
                    {memoryBook.coverPhoto ? (
                      <img 
                        src={memoryBook.coverPhoto.url} 
                        alt={memoryBook.title}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Heart className="h-8 w-8 text-gray-400" />
                      </div>
                    )}
                  </div>
                  <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1 truncate">
                    {memoryBook.title}
                  </h3>
                  <p className="text-xs text-stone">
                    {memoryBook.photoCount} photos
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Empty State */}
        {photos.length === 0 && adventures.length === 0 && memoryBooks.length === 0 && (
          <section className="px-4 py-8 text-center">
            <Camera className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-600 mb-2">Start Your Adventure</h3>
            <p className="text-gray-500 mb-4">Upload your first photos to begin organizing your outdoor memories.</p>
            <Button 
              onClick={() => setIsPhotoUploadOpen(true)}
              className="bg-forest hover:bg-forest/90 text-white"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Your First Photos
            </Button>
          </section>
        )}
      </main>

      {/* Floating Action Button */}
      <Button 
        onClick={() => setIsPhotoUploadOpen(true)}
        className="fixed bottom-24 right-4 bg-sunset hover:bg-sunset/90 text-white w-14 h-14 rounded-full shadow-lg flex items-center justify-center hover:shadow-xl transition-all transform hover:scale-110 z-40 p-0"
      >
        <Camera className="h-6 w-6" />
      </Button>

      {/* Modals */}
      {selectedPhoto && (
        <PhotoModal 
          photo={selectedPhoto} 
          onClose={() => setSelectedPhoto(null)} 
        />
      )}

      {isPhotoUploadOpen && (
        <PhotoUpload onClose={() => setIsPhotoUploadOpen(false)} />
      )}

      {isMemoryModalOpen && (
        <MemoryBookModal onClose={() => setIsMemoryModalOpen(false)} />
      )}
    </>
  );
}
