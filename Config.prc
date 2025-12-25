# Panda3D configuration for macOS OpenGL compatibility
# Disables all shader-dependent features

# Disable shader generation entirely
auto-shader false
basic-shaders-only true

# Force no shaders
hardware-animated-vertices false

# Disable all lighting effects that need shaders
support-stencil false

# Reduce framebuffer requirements  
framebuffer-stencil false
framebuffer-alpha true
framebuffer-multisample false
multisamples 0

# Disable problematic features
textures-power-2 none
textures-auto-power-2 false
hardware-point-sprites false

# Force simple rendering
depth-bits 24
color-bits 24

# Suppress shader error messages
notify-level-glgsg error
