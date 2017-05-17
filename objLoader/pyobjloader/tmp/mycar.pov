#version 3.6;

#include "colors.inc"

// Floor
plane { y, -0.25
    pigment { Gold }
    
}

// Something to light the front of the text
light_source { <0, 3, -0.1> color White*0.8}

// An extended area spotlight to backlight the letters
light_source {
   <0, 50, 100> color White

   // The spotlight parameters
   spotlight
   point_at <0, 0, -5>
   radius 6
   falloff 22

   // The extended area light paramaters
   area_light <6, 0, 0>, <0, 6, 0>, 9, 9
   adaptive 0
   jitter
}

background { color White }

camera {
  location <-1, 0.75, -1>
  look_at <0, 0, 0>
}

// light_source { <-4, 2, 0> color White shadowless}
light_source { <-2, 2, 0> color White shadowless}

#include "model.inc"

