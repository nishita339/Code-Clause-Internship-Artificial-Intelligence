import { useQuery } from "@tanstack/react-query";
import { Search, Filter, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import PhotoGrid from "@/components/photo-grid";
import PhotoModal from "@/components/photo-modal";
import CategoryFilter from "@/components/category-filter";
import AdventureCard from "@/components/adventure-card";
import type { Photo, Adventure } from "@shared/schema";

interface AdventureWithPhotos extends Adventure {
  photoCount: number;
  coverPhoto?: Photo | null;
}

export default function Explore() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const { data: photos = [] } = useQuery<Photo[]>({
    queryKey: selectedCategory === "all" 
      ? ['/api/photos'] 
      : ['/api/photos', { category: selectedCategory }],
  });

  const { data: adventures = [] } = useQuery<AdventureWithPhotos[]>({
    queryKey: ['/api/adventures'],
  });

  // Filter photos and adventures based on search query
  const filteredPhotos = photos.filter(photo => 
    photo.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    photo.location?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredAdventures = adventures.filter(adventure =>
    adventure.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    adventure.location.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <>
      {/* Header */}
      <header className="bg-forest text-white p-4 sticky top-0 z-50 shadow-lg">
        <div className="flex items-center space-x-3 mb-3">
          <Search className="text-2xl text-sage" />
          <h1 className="text-xl font-bold">Explore</h1>
        </div>
        
        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            type="text"
            placeholder="Search photos, adventures, locations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-white/10 border-white/20 text-white placeholder:text-white/60 focus:bg-white/20"
          />
        </div>
      </header>

      <main className="pb-20">
        {/* Category Filter */}
        <section className="p-4">
          <CategoryFilter 
            selectedCategory={selectedCategory}
            onCategoryChange={setSelectedCategory}
          />
        </section>

        {/* Search Results Summary */}
        {searchQuery && (
          <section className="px-4 pb-4">
            <div className="bg-sage/10 rounded-lg p-3">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Found {filteredPhotos.length} photos and {filteredAdventures.length} adventures
                {searchQuery && ` for "${searchQuery}"`}
              </p>
            </div>
          </section>
        )}

        {/* Adventures */}
        {filteredAdventures.length > 0 && (
          <section className="px-4 mb-6">
            <div className="flex items-center space-x-2 mb-3">
              <MapPin className="h-5 w-5 text-forest" />
              <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200">Adventures</h2>
            </div>
            
            <div className="space-y-3">
              {filteredAdventures.map((adventure) => (
                <AdventureCard key={adventure.id} adventure={adventure} />
              ))}
            </div>
          </section>
        )}

        {/* Photo Grid */}
        {filteredPhotos.length > 0 && (
          <section className="px-4 mb-6">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200">Photos</h2>
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="sm">
                  <Filter className="h-4 w-4 mr-1" />
                  Filter
                </Button>
              </div>
            </div>

            <PhotoGrid 
              photos={filteredPhotos} 
              onPhotoClick={setSelectedPhoto}
            />
          </section>
        )}

        {/* Empty State */}
        {filteredPhotos.length === 0 && filteredAdventures.length === 0 && (
          <section className="px-4 py-8 text-center">
            <Search className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-600 mb-2">
              {searchQuery ? "No results found" : "Start exploring"}
            </h3>
            <p className="text-gray-500 mb-4">
              {searchQuery 
                ? `Try searching for different keywords or browse by category.`
                : "Search for photos, adventures, or browse by category to discover your memories."
              }
            </p>
            {searchQuery && (
              <Button 
                onClick={() => setSearchQuery("")}
                variant="outline"
                className="border-forest text-forest hover:bg-forest hover:text-white"
              >
                Clear Search
              </Button>
            )}
          </section>
        )}
      </main>

      {/* Photo Modal */}
      {selectedPhoto && (
        <PhotoModal 
          photo={selectedPhoto} 
          onClose={() => setSelectedPhoto(null)} 
        />
      )}
    </>
  );
}
