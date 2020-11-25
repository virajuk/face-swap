from swap import SWAP

# img2 = 'images/dadario.jpg'
# img1 = 'images/a_wallis_.jpg'

img1 = 'images/bradley_cooper.jpg'
img2 = 'images/scarlet.jpg'

SWAP.enable_visualization()
swap = SWAP(img1, img2)

SWAP.enable_visualization()
swap.first_face_landmarks()

SWAP.enable_visualization()
swap.second_face_landmarks()

SWAP.enable_visualization(False)
swap.first_convex_hull()

SWAP.enable_visualization(False)
swap.second_convex_hull()

SWAP.enable_visualization(False)
swap.delaunay_triangle_1()

SWAP.enable_visualization(False)
swap.delaunay_triangle_2()

SWAP.enable_visualization(False)
swap.combine_triangle()

SWAP.enable_visualization()
swap.final_show()

swap.show_images()
