import imageio
import os

png_dir = 'frames/frames4'
images = []

# Get all png files in the directory
png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]

# Sort the files in order of their names (depends on how your files are named)
png_files.sort()

i = 0
for file_name in png_files:
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

# # Save the images as an animated gif
# imageio.mimsave('movie.gif', images)
# Save the images as an animated gif with custom frame duration
imageio.mimsave('movie_idea2.gif', images, duration = 1) # duration is the time spent on each image (in seconds)
