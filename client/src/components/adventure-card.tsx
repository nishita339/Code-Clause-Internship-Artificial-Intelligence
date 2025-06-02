import { MapPin, Calendar, Camera } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Adventure, Photo } from "@shared/schema";

interface AdventureWithPhotos extends Adventure {
  photoCount: number;
  coverPhoto?: Photo | null;
}

interface AdventureCardProps {
  adventure: AdventureWithPhotos;
}

export default function AdventureCard({ adventure }: AdventureCardProps) {
  const getCategoryColor = (category: string) => {
    switch (category.toLowerCase()) {
      case 'hiking':
        return 'bg-sage/20 text-sage';
      case 'camping':
        return 'bg-earth/20 text-earth';
      case 'climbing':
        return 'bg-stone/20 text-stone';
      case 'water-sports':
        return 'bg-sky/20 text-sky';
      default:
        return 'bg-forest/20 text-forest';
    }
  };

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow cursor-pointer">
      <CardContent className="p-0">
        <div className="flex">
          {/* Cover Image */}
          <div className="w-24 h-20 bg-gray-200 dark:bg-gray-700 flex-shrink-0">
            {adventure.coverPhoto ? (
              <img 
                src={adventure.coverPhoto.url} 
                alt={adventure.title}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <Camera className="h-6 w-6 text-gray-400" />
              </div>
            )}
          </div>
          
          {/* Content */}
          <div className="p-3 flex-1 min-w-0">
            <div className="flex items-start justify-between mb-2">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 truncate pr-2">
                {adventure.title}
              </h3>
              <Badge 
                variant="secondary" 
                className={`text-xs whitespace-nowrap ${getCategoryColor(adventure.category)}`}
              >
                {adventure.category.charAt(0).toUpperCase() + adventure.category.slice(1).replace('-', ' ')}
              </Badge>
            </div>
            
            <div className="flex items-center text-xs text-stone space-x-3 mb-1">
              <span className="flex items-center">
                <Calendar className="h-3 w-3 mr-1" />
                {new Date(adventure.date).toLocaleDateString()}
              </span>
              <span className="flex items-center">
                <Camera className="h-3 w-3 mr-1" />
                {adventure.photoCount} photos
              </span>
            </div>
            
            {adventure.location && (
              <div className="flex items-center text-xs text-stone">
                <MapPin className="h-3 w-3 mr-1" />
                <span className="truncate">{adventure.location}</span>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
