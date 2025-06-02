import { useQuery } from "@tanstack/react-query";
import { User, Camera, Heart, MapPin, Settings, Share } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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

export default function Profile() {
  const { data: stats } = useQuery<StatsData>({
    queryKey: ['/api/stats'],
  });

  const { data: photos = [] } = useQuery<Photo[]>({
    queryKey: ['/api/photos'],
  });

  const { data: adventures = [] } = useQuery<AdventureWithPhotos[]>({
    queryKey: ['/api/adventures'],
  });

  const { data: memoryBooks = [] } = useQuery<MemoryBookWithPhotos[]>({
    queryKey: ['/api/memory-books'],
  });

  const likedPhotos = photos.filter(photo => photo.isLiked);
  const recentAdventures = adventures.slice(0, 3);

  return (
    <>
      {/* Header */}
      <header className="bg-forest text-white p-4 flex items-center justify-between sticky top-0 z-50 shadow-lg">
        <div className="flex items-center space-x-3">
          <User className="text-2xl text-sage" />
          <h1 className="text-xl font-bold">Profile</h1>
        </div>
        <Button variant="ghost" size="icon" className="text-sage hover:text-white">
          <Settings className="h-5 w-5" />
        </Button>
      </header>

      <main className="pb-20">
        {/* Profile Info */}
        <section className="p-4 bg-gradient-to-r from-sage/20 to-sky/20">
          <div className="flex items-center space-x-4 mb-4">
            <div className="w-16 h-16 bg-forest rounded-full flex items-center justify-center">
              <User className="h-8 w-8 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-800 dark:text-gray-200">Adventure Explorer</h2>
              <p className="text-stone">Outdoor enthusiast since 2023</p>
            </div>
          </div>
          
          <div className="flex space-x-3">
            <Button size="sm" className="bg-forest hover:bg-forest/90 text-white">
              <Share className="h-4 w-4 mr-1" />
              Share Profile
            </Button>
            <Button size="sm" variant="outline" className="border-forest text-forest hover:bg-forest hover:text-white">
              Edit Profile
            </Button>
          </div>
        </section>

        {/* Stats Cards */}
        <section className="p-4">
          <div className="grid grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <Camera className="h-6 w-6 text-forest mx-auto mb-2" />
                <div className="text-2xl font-bold text-forest">
                  {stats?.totalPhotos || 0}
                </div>
                <div className="text-xs text-stone">Photos</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4 text-center">
                <MapPin className="h-6 w-6 text-sky mx-auto mb-2" />
                <div className="text-2xl font-bold text-sky">
                  {stats?.totalAdventures || 0}
                </div>
                <div className="text-xs text-stone">Adventures</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4 text-center">
                <Heart className="h-6 w-6 text-sunset mx-auto mb-2" />
                <div className="text-2xl font-bold text-sunset">
                  {stats?.totalMemories || 0}
                </div>
                <div className="text-xs text-stone">Memories</div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Recent Adventures */}
        {recentAdventures.length > 0 && (
          <section className="px-4 mb-6">
            <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200 mb-3">Recent Adventures</h2>
            <div className="space-y-3">
              {recentAdventures.map((adventure) => (
                <Card key={adventure.id} className="overflow-hidden">
                  <CardContent className="p-0">
                    <div className="flex">
                      <div className="w-20 h-20 bg-gray-200 dark:bg-gray-700 flex-shrink-0">
                        {adventure.coverPhoto ? (
                          <img 
                            src={adventure.coverPhoto.url} 
                            alt={adventure.title}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <MapPin className="h-6 w-6 text-gray-400" />
                          </div>
                        )}
                      </div>
                      <div className="p-3 flex-1">
                        <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">
                          {adventure.title}
                        </h3>
                        <div className="flex items-center text-xs text-stone space-x-4">
                          <span>{adventure.category}</span>
                          <span>{adventure.photoCount} photos</span>
                          <span>{new Date(adventure.date).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        )}

        {/* Liked Photos */}
        {likedPhotos.length > 0 && (
          <section className="px-4 mb-6">
            <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200 mb-3">Liked Photos</h2>
            <div className="grid grid-cols-3 gap-2">
              {likedPhotos.slice(0, 6).map((photo) => (
                <div key={photo.id} className="aspect-square rounded-lg overflow-hidden relative">
                  <img 
                    src={photo.url} 
                    alt={photo.title || "Liked photo"}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-1 right-1">
                    <Heart className="h-4 w-4 text-sunset fill-current" />
                  </div>
                </div>
              ))}
            </div>
            {likedPhotos.length > 6 && (
              <Button variant="link" className="w-full mt-2 text-forest">
                View All {likedPhotos.length} Liked Photos
              </Button>
            )}
          </section>
        )}

        {/* Memory Books Preview */}
        {memoryBooks.length > 0 && (
          <section className="px-4 mb-6">
            <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200 mb-3">Memory Books</h2>
            <div className="flex space-x-3 overflow-x-auto pb-2">
              {memoryBooks.slice(0, 4).map((memoryBook) => (
                <div key={memoryBook.id} className="flex-shrink-0 w-24">
                  <div className="w-24 h-32 bg-gray-200 dark:bg-gray-700 rounded-lg shadow-md mb-2 overflow-hidden">
                    {memoryBook.coverPhoto ? (
                      <img 
                        src={memoryBook.coverPhoto.url} 
                        alt={memoryBook.title}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Heart className="h-6 w-6 text-gray-400" />
                      </div>
                    )}
                  </div>
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate">
                    {memoryBook.title}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Settings */}
        <section className="px-4 mb-6">
          <h2 className="text-lg font-bold text-gray-800 dark:text-gray-200 mb-3">Settings</h2>
          <div className="space-y-2">
            <Button variant="ghost" className="w-full justify-start text-left p-3 h-auto">
              <Settings className="h-5 w-5 mr-3 text-stone" />
              <div>
                <div className="font-medium">App Settings</div>
                <div className="text-sm text-stone">Notifications, privacy, storage</div>
              </div>
            </Button>
            <Button variant="ghost" className="w-full justify-start text-left p-3 h-auto">
              <Share className="h-5 w-5 mr-3 text-stone" />
              <div>
                <div className="font-medium">Export Data</div>
                <div className="text-sm text-stone">Download your photos and memories</div>
              </div>
            </Button>
          </div>
        </section>
      </main>
    </>
  );
}
