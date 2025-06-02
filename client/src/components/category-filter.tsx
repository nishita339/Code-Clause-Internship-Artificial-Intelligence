import { Star, Backpack, Tent, Mountain, Waves, Camera, Tag } from "lucide-react";
import { Button } from "@/components/ui/button";
import { categories } from "@shared/schema";

interface CategoryFilterProps {
  selectedCategory: string;
  onCategoryChange: (category: string) => void;
}

const categoryIcons = {
  all: Star,
  hiking: Backpack,
  camping: Tent,
  climbing: Mountain,
  "water-sports": Waves,
  wildlife: Camera,
  landscape: Mountain,
  other: Tag,
};

const categoryLabels = {
  all: "All",
  hiking: "Backpack",
  camping: "Camping", 
  climbing: "Climbing",
  "water-sports": "Water Sports",
  wildlife: "Wildlife",
  landscape: "Landscape",
  other: "Other",
};

export default function CategoryFilter({ selectedCategory, onCategoryChange }: CategoryFilterProps) {
  const allCategories = ["all", ...categories];

  return (
    <div className="flex space-x-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-sage">
      {allCategories.map((category) => {
        const Icon = categoryIcons[category as keyof typeof categoryIcons];
        const isActive = selectedCategory === category;
        
        return (
          <Button
            key={category}
            onClick={() => onCategoryChange(category)}
            variant={isActive ? "default" : "secondary"}
            size="sm"
            className={`
              whitespace-nowrap transition-colors flex items-center space-x-1
              ${isActive 
                ? "bg-forest text-white hover:bg-forest/90" 
                : "bg-gray-200 dark:bg-gray-700 text-stone hover:bg-sage hover:text-white"
              }
            `}
          >
            <Icon className="h-4 w-4" />
            <span className="font-medium">
              {categoryLabels[category as keyof typeof categoryLabels]}
            </span>
          </Button>
        );
      })}
    </div>
  );
}
