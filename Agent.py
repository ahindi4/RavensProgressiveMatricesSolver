from PIL import Image,ImageChops
import numpy as np
import cv2

class Agent:
    def image_similarity(self, img1, img2):
        try:
            grayScale_img1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
            grayScale_img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)

            similarity_index = cv2.matchTemplate(grayScale_img1, grayScale_img2, cv2.TM_CCOEFF_NORMED)
            return similarity_index
        except Exception as e:
            print("Error in image similarity:", e)
            return None
        

    def image_similarity_2(self, img1, img2):
        
        Threshold = 0.98
        try:
            # Assuming img1 and img2 are loaded in BGR format, convert them to grayscale
            grayScale_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            grayScale_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Apply template matching
            similarity_index = cv2.matchTemplate(grayScale_img1, grayScale_img2, cv2.TM_CCOEFF_NORMED)
            max_similarity = np.max(similarity_index)  # Get the highest similarity coefficient

            
            if max_similarity >= Threshold:
                return max_similarity
            else:
                return None
            
        except Exception as e:
            print("Error in image similarity:", e)
            return None


    def flip_image_horizontally(self, image):
        """Flip the image horizontally."""
        return cv2.flip(image, 1)


    def flip_image_vertically(self, image):
        """Flip the image vertically."""
        return cv2.flip(image, 0)
 

    def check_image_reflection(self, img1, img2):
        # Check for horizontal reflection
        flipped_horizontally = self.flip_image_horizontally(img1)
        similarity_horizontal = self.image_similarity(flipped_horizontally, img2)
        

        if similarity_horizontal > 0.95:
            return 'Horizontal'

        flipped_vertically = self.flip_image_vertically(img1)
        similarity_vertical = self.image_similarity(flipped_vertically, img2)

        if similarity_vertical > 0.95:
            return 'Vertical'

        return None


    def find_best_match_for_c_horizontally(self, img_c, choice_images):
        # Flip the reference image vertically
        flipped_c_horizonatlly = self.flip_image_horizontally(img_c)
        
        max_similarity = 0
        best_option_index = -1
        closest_to_threshold = float('inf')

        for i, image_choice in enumerate(choice_images, start=1):
            similarity_score = self.image_similarity(flipped_c_horizonatlly, image_choice)

            if similarity_score >= 0.98:
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_option_index = i
                    
            elif abs(similarity_score - 0.98) < abs(closest_to_threshold - 0.98):
                closest_to_threshold = similarity_score
                closest_option_index = i

        if max_similarity < 0.98:
            return closest_option_index

        return best_option_index


    def find_perfect_match_for_c_vertically(self, img_input, choice_images):
        flipped_c_vertically = self.flip_image_vertically(img_input)
        
        max_similarity = 0
        best_option_index = -1
        closest_to_threshold = float('inf')
        

        # Iterate through choice images to find the best match
        for i, image_choice in enumerate(choice_images, start=1):
            # Calculate similarity score for the flipped version of C and each choice
            
            similarity_score = self.image_similarity(flipped_c_vertically, image_choice)


            if similarity_score >= 0.98:  # Consider adjusting the threshold based on your accuracy needs
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_option_index = i
                    
            elif abs(similarity_score - 0.98) < abs(closest_to_threshold - 0.98):
                closest_to_threshold = similarity_score
                closest_option_index = i

        if max_similarity < 0.98:
            # print("NO MATCH WAS FOUND BUT HERE IS THE CLOSEST option we got: Choice", closest_option_index)
            return closest_option_index

        return best_option_index


    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Use getRotationMatrix2D to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the image using warpAffine
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image
    

    def Image_Similarity_Testing(self, image1, image2):
        try:
            matrix_1 = np.array(image1).astype(float)
            matrix_2 = np.array(image2).astype(float)
            matrix_3 = np.subtract(matrix_1, matrix_2)
            pixel_difference = (np.abs(matrix_3) > 127).sum()

            no_of_pixels = image1.shape[0] * image1.shape[1]  # Use shape instead of size
            similarity = 1.0 - (pixel_difference / (no_of_pixels * 1.0))

            return similarity
        except Exception as e:
            print("Error in image similarity:", e)
            return None


    def check_image_rotation(self, img1, img2):
        # Define the angles to check
        angles = [90, 180, 270]
        best_similarity = 0
        best_angle = None
        
        for angle in angles:
            rotated_img = self.rotate_image(img1, angle)
            
            similarity_score = self.Image_Similarity_Testing(rotated_img, img2)
            # print(f"SIMILARITY SCORE FOR rotated image at {angle}: ", similarity_score)
            
            if similarity_score >= 0.90:
                best_similarity = similarity_score
                best_angle = angle

        return best_angle


    def rotate_selected_images(self, img1, choice_images, angle):
        best_similarity = 0
        best_choice_index = -1

        # Rotate img1 by the current angle
        rotated_img1 = self.rotate_image(img1, angle)
        
        # Iterate through choice images to find the best match
        for i, image_choice in enumerate(choice_images, start=1):
            similarity_score = self.Image_Similarity_Testing(rotated_img1, image_choice)
            
            # Check if the similarity is above the best found so far
            if similarity_score > best_similarity:
                best_similarity = similarity_score
                best_choice_index = i

        if best_choice_index > 0:
            return best_choice_index
        else:
            return None


    def count_black_pixels(self, image):
        try:

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            black_pixels = cv2.countNonZero(binary)
            
            return black_pixels
        except Exception as e:
            print("Error in counting black pixels:", e)
            return None

    
    def Solve(self, problem):

        if problem.problemType == "2x2":
            image_a = cv2.imread(problem.figures["A"].visualFilename)
            image_b = cv2.imread(problem.figures["B"].visualFilename)
            image_c = cv2.imread(problem.figures["C"].visualFilename)

            image_choices = [cv2.imread(problem.figures[str(i)].visualFilename) for i in range(1, 7)]

            identical_check_a_b = self.image_similarity_2(image_a, image_b)
            identical_check_a_c = self.image_similarity_2(image_a, image_c)
            
            reflection_a_b = self.check_image_reflection(image_a, image_b)
            reflection_a_c = self.check_image_reflection(image_a, image_c)
            reflection_b_c = self.check_image_reflection(image_b, image_c)
            
            #Variable for rotation check
            rotation_angle_a_b = self.check_image_rotation(image_a, image_b)
            rotation_angle_a_c = self.check_image_rotation(image_a, image_c)
            rotation_angle_b_c = self.check_image_rotation(image_b, image_c) 
            
            #Image Simliarties between A and B
            similarity_a_b = self.image_similarity(image_a, image_b)
            
            if identical_check_a_b != None:
                max_similarity = 0
                best_option_index = -1
                
                for i, image_choice in enumerate(image_choices, start=1):
                    similarity_c_i = self.image_similarity(image_c, image_choice)
                    if similarity_c_i >= 0.97:
                        if similarity_c_i > max_similarity:
                            max_similarity = similarity_c_i
                            best_option_index = i  # Adjust index for 1-based option numbering
                
                return best_option_index
            
            elif identical_check_a_c != None:
                max_similarity = 0
                best_option_index = -1
                
                for i, image_choice in enumerate(image_choices, start=1):
                    similarity_b_i = self.image_similarity(image_b, image_choice)
                    if similarity_b_i >= 0.97:
                        if similarity_b_i > max_similarity:
                            max_similarity = similarity_b_i
                            best_option_index = i   # Adjust index for 1-based option numbering
                
                return best_option_index
            
            elif reflection_a_b:
                # print("Reflection between A and B")
                if reflection_a_b == "Horizontal":
                    best_match_index = self.find_best_match_for_c_horizontally(image_c, image_choices)
                    if best_match_index != -1:
                        # print(f"FOUND A MATCH BETWEEN C and choice {best_match_index} in problem {problem.name}")
                        return best_match_index
                    
                elif reflection_a_b == "Vertical":
                    best_match_index = self.find_perfect_match_for_c_vertically (image_c, image_choices)
                    if best_match_index != -1:
                        # print(f"FOUND A MATCH BETWEEN C and choice {best_match_index} in problem {problem.name}")
                        return best_match_index
                    

            elif reflection_a_c:
                # print("Reflection between A and C")
                # print("Reflection between A and C type: ", reflection_a_c)
                if reflection_a_c == "Horizontal":
                    best_match_index = self.find_best_match_for_c_horizontally(image_b, image_choices)
                    if best_match_index != -1:
                        # print(f"FOUND A MATCH BETWEEN C and choice {best_match_index} in problem {problem.name}")
                        return best_match_index
                    
                elif reflection_a_c == "Vertical":
                    best_match_index = self.find_perfect_match_for_c_vertically(image_b, image_choices)
                    if best_match_index != -1:
                        # print(f"FOUND A MATCH BETWEEN C and choice {best_match_index} in problem {problem.name}")
                        return best_match_index

            elif reflection_b_c:
                # print("Reflection between B and C")
                if reflection_b_c == "Horizontal":
                    best_match_index = self.find_best_match_for_c_horizontally(image_c, image_choices)
                    if best_match_index != -1:
                        # print(f"FOUND A MATCH BETWEEN C and choice {best_match_index} in problem {problem.name}")
                        return best_match_index
                    
                elif reflection_b_c == "Vertical":
                    best_match_index = self.find_perfect_match_for_c_vertically(image_a, image_choices)
                    if best_match_index > -1:
                        # print(f"FOUND A MATCH BETWEEN C and choice {best_match_index} in problem {problem.name}")
                        return best_match_index
                    else:
                        return -1
                    
            elif rotation_angle_a_b:
                # print(f"Image A is rotated by {rotation_angle_a_b} degrees to match Image B.")
                best_match_index = self.rotate_selected_images(image_c, image_choices, rotation_angle_a_b)
                # print(f"The image choice is {best_match_index}")
                if best_match_index != None:
                    return best_match_index
                else:
                    return -1

            elif rotation_angle_a_c:
                # print(f"Image A is rotated by {rotation_angle_a_c} degrees to match Image C.")
                best_match_index = self.rotate_selected_images(image_b, image_choices, rotation_angle_a_c)
                # print(f"The image choice is {best_match_index}")
                if best_match_index != None:
                    return best_match_index
                else:
                    return -1
            elif rotation_angle_b_c:
                # print(f"Image B is rotated by {rotation_angle_b_c} degrees to match Image C.")
                best_match_index = self.rotate_selected_images(image_a, image_choices, rotation_angle_b_c)
                # print(f"The image choice is {best_match_index}")
                if best_match_index != None:
                    return best_match_index
                else:
                    return -1
                
            elif similarity_a_b > 0:
                if similarity_a_b is None:
                    return -1

                min_difference = float('inf')
                best_choice_index = -1

                for i, image_choice in enumerate(image_choices, start=1):
                    similarity_c_i = self.image_similarity(image_c, image_choice)
                    if similarity_c_i is None:
                        continue

                    difference = abs(similarity_c_i - similarity_a_b)
                    if difference < min_difference:
                        min_difference = difference
                        best_choice_index = i

                return best_choice_index
                
            
            else:
                print("None")    
                return -1                        
            
        elif problem.problemType == "3x3":

            image_a = cv2.imread(problem.figures["A"].visualFilename)                
            image_b = cv2.imread(problem.figures["B"].visualFilename)
            image_c = cv2.imread(problem.figures["C"].visualFilename)
            image_d = cv2.imread(problem.figures["D"].visualFilename)
            image_e = cv2.imread(problem.figures["E"].visualFilename)
            image_f = cv2.imread(problem.figures["F"].visualFilename)
            image_g = cv2.imread(problem.figures["G"].visualFilename)
            image_h = cv2.imread(problem.figures["H"].visualFilename)
            
            image_a_pixel_count = self.count_black_pixels(image_a)
            image_b_pixel_count = self.count_black_pixels(image_b)
            image_c_pixel_count = self.count_black_pixels(image_c)
            image_d_pixel_count = self.count_black_pixels(image_d)
            image_e_pixel_count = self.count_black_pixels(image_e)
            image_f_pixel_count = self.count_black_pixels(image_f)
            image_g_pixel_count = self.count_black_pixels(image_g)
            image_h_pixel_count = self.count_black_pixels(image_h)
            
            if (image_a_pixel_count == image_b_pixel_count == image_c_pixel_count):
                for i in range(1, 9):
                    image_i = cv2.imread(problem.figures[str(i)].visualFilename)
                    black_pixels_i = self.count_black_pixels(image_i)
                    if image_h_pixel_count == black_pixels_i:
                        # print("FOUND SIMLIAR IMAGES IN PROBLEM: ", problem.name, " the choice is: ", i)
                        return i
            elif image_a_pixel_count == image_e_pixel_count:
                # If image A and image E have equal pixel counts
                # Find the image choice with the closest pixel count to image E
                min_diff = float('inf')
                best_choice = None
                for i in range(1, 9):
                    image_i = cv2.imread(problem.figures[str(i)].visualFilename)
                    black_pixels_i = self.count_black_pixels(image_i)
                    diff = abs(image_e_pixel_count - black_pixels_i)
                    if diff < min_diff:
                        min_diff = diff
                        best_choice = i
                if best_choice is not None:
                    # print("FOUND SIMILAR IMAGES IN PROBLEM:", problem.name, "the choice is:", best_choice)
                    return best_choice
                
            
            images_a_h = [image_a, image_b, image_c, image_d, image_e, image_f, image_g, image_h]

            image_choices = [cv2.imread(problem.figures[str(i)].visualFilename) for i in range(1, 9)]
            
            min_differences = {}
            unique_choices = set()

            for i in range(1, 9):
                unique_choices.add(i)

            for idx, img in enumerate(images_a_h, start=1):
                min_diff = float('inf')
                min_choice = None
                for i, img_choice in enumerate(image_choices, start=1):
                    diff = abs(self.count_black_pixels(img) - self.count_black_pixels(img_choice))
                    if diff < min_diff:
                        min_diff = diff
                        min_choice = i
                min_differences[f"Image_{chr(ord('A') + idx - 1)}"] = (min_choice, min_diff)

            for img, (min_choice, min_diff) in min_differences.items():
                
                # print(f"Minimum difference for {img} is with image choice {min_choice} with difference {min_diff}")
                if min_diff < 62:
                    unique_choices.discard(min_choice)
            
            unique_choices = list(unique_choices)
            
            if len(unique_choices) == 1:
                return unique_choices[0]

            black_pixels_h = np.count_nonzero(image_h == 0)
            ratio_h_g = np.count_nonzero(image_h == 0) / np.count_nonzero(image_g == 0)

            unique_ratios = []

            for unique_choice in unique_choices:
                image_choice = cv2.imread(problem.figures[str(unique_choice)].visualFilename)
                black_pixels_choice = np.count_nonzero(image_choice == 0)
                ratio_choice_h = black_pixels_choice / black_pixels_h
                unique_ratios.append((unique_choice, ratio_choice_h))

            closest_match = min(unique_ratios, key=lambda x: abs(x[1] - ratio_h_g))

            return closest_match[0]
