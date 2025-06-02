import { useState } from "react";
import { X, Book, Heart, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface MemoryBookModalProps {
  onClose: () => void;
}

export default function MemoryBookModal({ onClose }: MemoryBookModalProps) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const createMutation = useMutation({
    mutationFn: async (data: { title: string; description?: string }) => {
      return apiRequest('POST', '/api/memory-books', data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/memory-books'] });
      queryClient.invalidateQueries({ queryKey: ['/api/stats'] });
      toast({
        title: "Memory book created!",
        description: "Your new memory book is ready for photos",
      });
      onClose();
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to create memory book",
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!title.trim()) {
      toast({
        title: "Title required",
        description: "Please enter a title for your memory book",
        variant: "destructive",
      });
      return;
    }

    createMutation.mutate({
      title: title.trim(),
      description: description.trim() || undefined,
    });
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
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 w-full max-w-md mx-4 animate-slide-up">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Heart className="h-6 w-6 text-sunset" />
            <h2 className="text-xl font-bold text-gray-800 dark:text-gray-200">Create Memory Book</h2>
          </div>
          <Button
            onClick={onClose}
            variant="ghost"
            size="icon"
            className="text-gray-500 hover:text-gray-700"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="title" className="flex items-center text-sm font-medium mb-1">
              <Book className="h-4 w-4 mr-1" />
              Title *
            </Label>
            <Input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Summer Adventures 2024"
              className="w-full"
              maxLength={100}
            />
          </div>

          <div>
            <Label htmlFor="description" className="text-sm font-medium mb-1 block">
              Description
            </Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe this collection of memories..."
              className="w-full"
              rows={3}
              maxLength={500}
            />
            <p className="text-xs text-stone mt-1">
              {description.length}/500 characters
            </p>
          </div>

          <div className="bg-sage/10 rounded-lg p-3">
            <div className="flex items-start space-x-2">
              <Plus className="h-5 w-5 text-forest mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                  Add photos later
                </p>
                <p className="text-xs text-stone">
                  You can add photos to your memory book after creating it by selecting photos and choosing "Add to Memory"
                </p>
              </div>
            </div>
          </div>

          <div className="flex space-x-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              className="flex-1 bg-sunset hover:bg-sunset/90 text-white"
              disabled={createMutation.isPending || !title.trim()}
            >
              {createMutation.isPending ? (
                "Creating..."
              ) : (
                <>
                  <Heart className="h-4 w-4 mr-2" />
                  Create Memory Book
                </>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
