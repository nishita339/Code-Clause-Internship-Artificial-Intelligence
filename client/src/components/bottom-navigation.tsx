import { Home, Compass, Heart, User, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLocation } from "wouter";
import { useState } from "react";
import PhotoUpload from "./photo-upload";

export default function BottomNavigation() {
  const [location, setLocation] = useLocation();
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const navItems = [
    { path: "/", icon: Home, label: "Home" },
    { path: "/explore", icon: Compass, label: "Explore" },
    { path: "/memories", icon: Heart, label: "Memories" },
    { path: "/profile", icon: User, label: "Profile" },
  ];

  const isActive = (path: string) => location === path;

  return (
    <>
      <nav className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 z-50">
        <div className="max-w-md mx-auto">
          <div className="flex items-center justify-around py-2">
            {navItems.map(({ path, icon: Icon, label }) => (
              <Button
                key={path}
                onClick={() => setLocation(path)}
                variant="ghost"
                className={`flex flex-col items-center py-2 px-4 ${
                  isActive(path) 
                    ? "text-forest" 
                    : "text-stone hover:text-forest"
                } transition-colors h-auto`}
              >
                <Icon className={`h-5 w-5 mb-1 ${isActive(path) ? "fill-current" : ""}`} />
                <span className="text-xs font-medium">{label}</span>
              </Button>
            ))}
            
            {/* Center Add Button */}
            <Button
              onClick={() => setIsUploadOpen(true)}
              className="flex flex-col items-center py-2 px-4 bg-forest hover:bg-forest/90 text-white rounded-full -mt-4 shadow-lg transition-all transform hover:scale-110 h-auto"
            >
              <Plus className="h-6 w-6" />
            </Button>
          </div>
        </div>
      </nav>

      {/* Photo Upload Modal */}
      {isUploadOpen && (
        <PhotoUpload onClose={() => setIsUploadOpen(false)} />
      )}
    </>
  );
}
