import { useState } from "react";
import { X, Heart, Edit, Share, Plus, MapPin, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import type { Photo } from "@shared/schema";

interface PhotoModalProps {
  photo: Photo;
  onClose: () => void;
}

export default function PhotoModal({ photo, onClose }: PhotoModalProps) {
  const [isLiked, setIsLiked] = useState(photo.isLiked);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const likeMutation = useMutation({
    mutationFn: async (liked: boolean) => {
      return apiRequest('PATCH', `/api/photos/${photo.id}`, { isLiked: liked });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/photos'] });
      toast({
        title: isLiked ? "Photo liked!" : "Photo unliked",
        description: isLiked ? "Added to your favorites" : "Removed from favorites",
      });
    },
    onError: () => {
      setIsLiked(!isLiked); // Revert on error
      toast({
        title: "Error",
        description: "Failed to update photo",
        variant: "destructive",
      });
    },
  });

  const handleLikeToggle = () => {
    const newLikedState = !isLiked;
    setIsLiked(newLikedState);
    likeMutation.mutate(newLikedState);
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center animate-fade-in"
      onClick={handleBackdropClick}
    >
      <div className="relative w-full h-full flex items-center justify-center p-4">
        <Button
          onClick={onClose}
          variant="ghost"
          size="icon"
          className="absolute top-4 right-4 text-white hover:bg-white/20 z-10"
        >
          <X className="h-6 w-6" />
        </Button>
        
        <div className="relative w-full max-w-lg animate-slide-up">
          {/* Main Photo */}
          <div className="relative">
            <img 
              src={photo.url} 
              alt={photo.title || "Adventure photo"}
              className="w-full rounded-lg max-h-[70vh] object-contain"
            />
          </div>
          
          {/* Photo Details Overlay */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 text-white rounded-b-lg">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-lg">
                {photo.title || "Untitled"}
              </h3>
              <Button
                onClick={handleLikeToggle}
                variant="ghost"
                size="icon"
                className="text-white hover:bg-white/20"
                disabled={likeMutation.isPending}
              >
                <Heart 
                  className={`h-6 w-6 transition-colors ${
                    isLiked 
                      ? 'text-sunset fill-current animate-bounce-gentle' 
                      : 'text-white'
                  }`} 
                />
              </Button>
            </div>
            
            <div className="flex items-center text-sm space-x-4 mb-3">
              {photo.location && (
                <span className="flex items-center">
                  <MapPin className="h-4 w-4 mr-1" />
                  {photo.location}
                </span>
              )}
              <span className="flex items-center">
                <Calendar className="h-4 w-4 mr-1" />
                {new Date(photo.uploadedAt!).toLocaleDateString()}
              </span>
            </div>
            
            <div className="flex space-x-3">
              <Button
                size="sm"
                className="bg-forest/80 hover:bg-forest text-white"
              >
                <Edit className="h-4 w-4 mr-2" />
                Edit
              </Button>
              <Button
                size="sm"
                className="bg-sky/80 hover:bg-sky text-white"
              >
                <Share className="h-4 w-4 mr-2" />
                Share
              </Button>
              <Button
                size="sm"
                className="bg-sunset/80 hover:bg-sunset text-white"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add to Memory
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
