from utils.swap import SWAP

# both images needs to be exact same resolution

# img1 = 'images/anna.jpg'
# img2 = 'images/lawrence-2-scaled.jpg'

# img1 = 'images/chris-hems.jpg'
# img2 = 'images/tom-2.jpg'

img1 = 'images/annabelle-wallis-600.jpg'
img2 = 'images/elsa.jpg'

swap = SWAP(img1, img2)

swap.swap()
# swap.my()
swap.show_images()
