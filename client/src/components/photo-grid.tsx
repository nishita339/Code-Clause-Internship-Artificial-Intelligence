import { Heart } from "lucide-react";
import type { Photo } from "@shared/schema";

interface PhotoGridProps {
  photos: Photo[];
  onPhotoClick: (photo: Photo) => void;
}

export default function PhotoGrid({ photos, onPhotoClick }: PhotoGridProps) {
  if (photos.length === 0) {
    return null;
  }

  return (
    <div className="grid grid-cols-2 gap-3">
      {photos.map((photo) => (
        <div 
          key={photo.id}
          onClick={() => onPhotoClick(photo)}
          className="relative rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow aspect-square cursor-pointer photo-item"
        >
          <img 
            src={photo.url} 
            alt={photo.title || "Adventure photo"}
            className="w-full h-full object-cover"
            loading="lazy"
          />
          
          {/* Like indicator */}
          <div className="absolute top-2 right-2">
            <Heart 
              className={`h-4 w-4 ${
                photo.isLiked 
                  ? 'text-sunset fill-current' 
                  : 'text-white opacity-80'
              }`} 
            />
          </div>
          
          {/* Location overlay */}
          {photo.location && (
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-2">
              <div className="text-white text-xs truncate">
                {photo.location}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
