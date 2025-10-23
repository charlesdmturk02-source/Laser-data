import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.special import jn
def read_diffraction_data(filename, start_string):
    """
    Function to extract every piece of data from within the chose text file which is stored under the variable 'filename'. 
    
    The variables wavelength, distance_to_screen and axis_units are initialised as empty.
    The data file is opened and 3 specific stings of text are searched for on a line by line basis and the quantities next to them are stored under the specific variables.
    Then the position and intensity data is stored in a 2D array in the case of intensity and 2 1D arrays for the X and Y positions, the code searches for the string '&END' before it starts reading data. 
    If the axis_units are in terms of mm then they need to be converted in terms of theta and phi so there is a conditional statement which takes care of that
    Finally the quantities theta, phi, intensity, wavelength, distance_to_screen, axis_units are returned to be used later on in the code.
    """
    # Initialize variables to store extracted quantities
    wavelength = None
    distance_to_screen = None
    axis_units = None

    # Open the file and read line by line until start_string is found
    with open(filename, 'r') as file:
        for line in file:
            if start_string in line:
                break
            # Check if the line contains any of the quantities we are interested in
            if "Wavelength (nm)=" in line:
                wavelength = float(line.split('=')[1]) * 1e-9
            elif "Distance to Screen" in line:
                distance_to_screen = float(line.split('=')[1])
            elif "Horizontal, Vertical units" in line:
                axis_units = str(line.split('=')[1][:-1])

        # Once start_string is found, read the remaining data using np.loadtxt
        data = np.loadtxt(file)
        
        if axis_units == 'mm':
            theta = np.arctan((data[0, 1:]/1000) / distance_to_screen)
            phi = np.arctan((data[1:, 0]/1000) / distance_to_screen)
        else:
            theta = np.arcsin(data[0, 1:]) # Extract x values from the first row, excluding the first element
            phi = np.arcsin(data[1:, 0]) # Extract y values from the first column, excluding the first element
            
            

    # Extract intensity values
    intensity_found = data[1:, 1:]
    intensity = np.array(intensity_found)  

    return theta, phi, intensity, wavelength, distance_to_screen, axis_units

def calculate_k(wavelength):
    """
    Function which calculates the value of k which is the wave number.
    
    The calculation is simple since k is just equal to (2 * pi)/wavelength 
    This is important since the wavelength is variable so a calculation must be performed
    
    This variable is returned as k
    """
    k = 2 * np.pi / wavelength
    return k

def calculate_diagonal_angle(theta, phi):
    """
    This function calculates the diagonal angle by combining the terms phi and theta.
    
    It then returns a value of this angle under the name diagonal_angle and is saved as a 1D array
    """
    diagonal_angle = np.sqrt(theta**2 + phi**2)
    return diagonal_angle

def extract_intensity_slice(intensity, coordinates, coordinate_value, axis):
    """
    This function extract an intensity slices from the 2D intensity array along the specified axis at the given coordinate value.
    The intenity array is loaded. It's a 2D array so either an X or Y position can be chose and a data slice can be taken.
    At the point where the X/Y co-ordinate is = 0 index of either the phi/theta array is taken
    Then at that index position a slice is taken at that index position in the intensity array

    The intenisty_slice is returned and can be used to take an intesity slice at any point essentially, this is called upon in the main function
    """
    # Find the index corresponding to the specified coordinate value
    index = np.argmin(np.abs(coordinates - coordinate_value))
    
    # Extract the intensity slice along the specified axis at the specified coordinate value
    if axis == 0:
        intensity_slice = intensity[index, :]
    elif axis == 1:
        intensity_slice = intensity[:, index]
    else:
        raise ValueError("Axis value must be 0 or 1.")
    
    return intensity_slice

def extract_diagonal_slice(intensity, reverse=False):
    """
    This function extracts the diagonal slice from the intensity array. 
    If reverse is False, it starts from the top-left corner and moves to the bottom-right corner. 
    If reverse is True, it starts from the top-right corner and moves to the bottom-left corner. 
    The length of the diagonal slice is determined by the minimum of the number of rows and columns in the intensity array.
    """
    # Get the dimensions of the intensity array
    rows, cols = intensity.shape
    # Initialize an empty list to store the diagonal slice
    diagonal_slice = []
    # Iterate over the array and select elements based on the direction of the diagonal slice
    if reverse:
        for i in range(min(rows, cols)):
            diagonal_slice.append(intensity[i, cols - 1 - i])
    else:
        for i in range(min(rows, cols)):
            diagonal_slice.append(intensity[i, i])
    return diagonal_slice

def W_I0_model(theta, W, I0, k):
    
    """
    This function is a modelling function which is used to determine both the maximum intensity of the diffraction pattern and the 'width' of the aperture
    
    It uses the equation discussed in the task assignment and returns a modelled version using the curve_fit import from scipy
    The way this function is called upon can be seen in the main function 
    Once this function is called it returns a value for the width of the aperture and the maximum intensity along with associated uncertainties
    
    """
    return I0 * np.sinc(k * W * np.sin(theta) / 2)**2

def L_model(phi, L, I0, k):
    """
    This function is used to calculate the value for L which is the height of the diffractionn aperture 
    
    Essentially uses the same eqaution as in the width model, however, the axis is now in terms of phi instead of theta 
    It also uses a slice taken starting from the phi position of 0
    Now that max_intensity is known it doesnt need to be calculated so only L is found
    
    Once this function is called it returns a value for the height of the aperture along with an associated uncertainty
    
    """

    return I0 * np.sinc(k * L * np.sin(phi) / 2)**2

def D_model(theta, D, I0, k):
    
    """
    This function is the modelling for finding the side length of a diamond aperture 
    
    Like the other fucntions above it trys to fit the the function to the data slice at either theta = 0 or phi = 0, in this case it's theta = 0
    When called upon a value for D is returned along with an associated uncertainty, the value D corresponds with the side length of the diamond
    """
    return I0 * np.sinc((k * D * np.sin(theta))/ (2 * np.sqrt(2)))**4

def C_model(I0, C, theta, sin_theta, k):
    """
    This is the final modelling function which is used in order to find the value of the diameter of the circular aperture.
    
    It models to the Intensity function using the bessel function which is imported from scipy 
    
    After being called upon it returns C which corresponds to the diameter of a circle along with an uncertainty
    """
    # Check if sin(theta) is close to zero
    sin_theta = np.sin(theta) 

# Replace zeros with a small epsilon value
    sin_theta = np.where(sin_theta == 0, np.finfo(float).eps, sin_theta)
    
    # Calculate the model with modified sin(theta)
    intensity = I0 * ((2 * jn(1, np.pi * k * C * sin_theta)) / (np.pi * k * C * sin_theta)) ** 2
    return intensity

def calculate_area_and_uncertainty(diameter, angular_diameter, width, height, side_length, diameter_error, width_error, height_error, side_length_error):
    """
    This function has quite a few functions: 
        The first is to determine the shape of the diffraction aperture.
        Then once the shape has been determined the specific dimenstions are saved as either dimension1 or dimension2 in order to be returned later in the results dict
        Then using the specific method the area is calculated along with the propagated uncertainty 
    Area_uncertainty, dimension1, dimension2, dimension1_err and dimesion2_err are initioalised and stored as zero
    Then the conditional statement goes through a check in order to find out what the shape is.
    if the diameter taken at theta = 0 and at the diagonal (angular_diameter) are the same then that means that the shape is a circle, only the first 5 characters are checked since noisy data can return slightly different results
    elif the width and height are different then it must be a rectangle since a square and diamond would have equal width and height 
    elif the diameter is more than the angular diameter then that means that it's a diamond since the corner to corner length is more than the side to side width
    If none of these are satisfied then that means that it must be a square
    Once the shape is determined then the method the area can be calculated using the various methods along with an uncertainty 
    Then finally the appropriate dimensions are stored in the correct variable dimension 1 or 2 in order to be stored in the final dictionary 
    """
    Area_uncertainty = None  # Initialize Area related variables 
    dimension1 = None
    dimension2 = None
    dimension1_err = None
    dimension2_err = None
    if str(angular_diameter)[:4] == str(diameter)[:4]:
        shape = "circle"
        dimension1 = diameter
        dimension2 = diameter
        dimension1_err = diameter_error
        dimension2_err = diameter_error
        Area = np.pi * (diameter / 2) ** 2
        # Propagate uncertainty for circular area
        Area_uncertainty = np.pi * (diameter / 2) * (diameter_error / 2)  # Using error propagation formula

    elif str(width)[:2] != str(height)[:2]:
        shape = "rectangle"
        dimension1 = width
        dimension2 = height
        dimension1_err = width_error
        dimension2_err = height_error
        Area = width * height
        # Propagate uncertainty for rectangle area
        Area_uncertainty = np.sqrt((width_error * height) ** 2 + (width * height_error) ** 2)  # Using error propagation formula

    elif diameter > angular_diameter:
        shape = "diamond"
        dimension1 = side_length
        dimension2 = side_length
        dimension1_err = side_length_error
        dimension2_err = side_length_error
        Area = side_length ** 2
        # Propagate uncertainty for diamond area
        Area_uncertainty = 2 * np.sqrt((side_length_error * side_length) ** 2)  # Using error propagation formula

    else:
        shape = "square"
        dimension1 = width
        dimension2 = height
        dimension1_err = width_error
        dimension2_err = height_error
        Area = width ** 2
        # Propagate uncertainty for square area
        Area_uncertainty = 2 * np.sqrt((width_error * width) ** 2)
    return shape, Area, Area_uncertainty, dimension1, dimension2, dimension1_err, dimension2_err



def calculate_power_and_uncertainty(max_intensity, Area, max_intenisty_error, Area_uncertainty):
    """
    This function is used to find the power of the laser being used in the experiment once again caculating a propagated uncertainty along with the absolute value
    It creates the variable Power which is satisfied by the relationship maximum intensity/ the area of the diffraction aperture
    
    It returns both the variables Power and Power_uncertainty 
    """
    Power = max_intensity / Area
    Power_uncertainty = Power * np.sqrt((max_intenisty_error / max_intensity) ** 2 + (Area_uncertainty / Area) ** 2)
    
    return Power, Power_uncertainty

def apply_gamma_correction(intensity, gamma):
    """
    This function applies gamma correction to the intensity data. Gamma correction is a non linear operation used to make the plots
    produced a lot more visable and easier to read. It makes the plots more visible by adjusting the intensity values.
    """
    min_intensity = np.min(intensity)
    intensity_shifted = intensity - min_intensity + 1e-10
    intensity_corrected = np.power(intensity_shifted, gamma)
    return intensity_corrected

def plot_diffraction_data(phi, theta, intensity, gamma=0.5, sigma=1.5, threshold=1.9e-4):
    """
    This code takes diffraction data, applies gamma correction to enhance visibility, smoothes the data using a Gaussian filter, 
    and optionally applies thresholding. 
    
    Then it plots the resulting diffraction pattern with proper labeling and a colorbar. 
    """
    # Apply gamma correction to intensity data
    adjusted_intensity = apply_gamma_correction(intensity, gamma)
    
    # Apply Gaussian smoothing
    smoothed_intensity = gaussian_filter(adjusted_intensity, sigma=sigma)
    
    # Apply thresholding 
    if threshold is not None:
        smoothed_intensity[smoothed_intensity < threshold] = threshold
    
    # Scale intensity data
    scaled_intensity = smoothed_intensity * 1e5  
    
    # Transpose the intensity array
    scaled_intensity = np.transpose(scaled_intensity)
    
    # Plot
    plt.imshow(scaled_intensity, cmap='hot', extent=[theta.min(), theta.max(), phi.min(), phi.max()], origin='lower')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Phi (radians)')
    plt.title('Diffraction Pattern Data py22ct')
    plt.colorbar(label='Intensity', format='%.0f%%')
    plt.show()


def plot_intensity_slices(theta, diagonal_slice, diagonal_angle, theta_0_slice, phi, phi_0_slice, k):
    """
    This function plots the horizontal and verticle slices of intensity and overlays the shape fittings for each shape
    
    It's not really possible to plot one for rectangles since it uses the same method as squares but the peak for 
    rectangles will be different sizes depending on the orientation of the rectangle.
    """
    plt.plot(theta, theta_0_slice, marker='x', markersize=3, linestyle='none', color='red')
    plt.plot(theta, W_I0_model(theta, *popt_W_I0, k), label='Square', linewidth=2, color = 'black')
    plt.plot(theta, D_model(theta, *popt_D, max_intensity, k), label='Diamond', color='green', linewidth=2) 
    plt.plot(theta, C_model(max_intensity, *popt_C, theta, np.sin(theta), k), label='Circle', color='orange', linewidth=2)
    plt.xlabel('Sin(Phi)')
    plt.ylabel('Intensity')
    plt.title('Intensity Slice at Sin(Theta) = 0')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(phi, phi_0_slice, marker='x', markersize=3, linestyle='none', color='red')
    plt.plot(theta, W_I0_model(phi, *popt_W_I0, k), label='Square', linewidth=2, color='black')
    plt.plot(phi, D_model(phi, *popt_D, max_intensity, k), label='Diamond', color='green', linewidth=2) 
    plt.plot(phi, C_model(max_intensity, *popt_C, phi, np.sin(phi), k), label='Circle', color='orange', linewidth=2)
    plt.xlabel('Sin(Theta)')
    plt.ylabel('Intensity')
    plt.title('Intensity Slice at Sin(Phi) = 0')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    
    """
    This function serves as the main entry point for the analysis of diffraction data and subsequent calculations.

    It performs the following tasks:
    1. Reads diffraction data from a specified file.
    2. Calculates parameters such as wavevector (k), initial guesses, and diagonal angle.
    3. Extracts intensity slices from the diffraction data.
    4. Fits models to the intensity slices to determine various parameters like width, height, side length, and diameter.
    5. Calculates uncertainties for the determined parameters.
    6. Computes the shape, area, area uncertainty, dimensions, and their uncertainties.
    7. Calculates power and its uncertainty based on the determined maximum intensity and area.
    8. Plots the diffraction data and intensity slices.
    """

    
    filename = "assessment_data_py22ct.dat"
    global power, power_uncertainty, shape, dimension1, dimension2, dimension1_err, dimension2_err, popt_W_I0, k, popt_L, max_intensity, popt_D, popt_C, sin_theta

    theta, phi, intensity, wavelength, distance_to_screen, axis_units = read_diffraction_data(filename, "&END")
    k = calculate_k(wavelength)
    initial_guess = 3e-5
    initial_guess_I0 = 5e-8
    diagonal_angle = calculate_diagonal_angle(theta, phi)

    x_coordinate = 0
    y_coordinate = 0
    theta_0_slice = extract_intensity_slice(intensity, theta, x_coordinate, axis=0)
    phi_0_slice = extract_intensity_slice(intensity, phi, y_coordinate, axis=1)
    diagonal_slice = extract_diagonal_slice(intensity)

    initial_guess_W = [initial_guess, initial_guess_I0]
    popt_W_I0, pcov_W_I0 = curve_fit(lambda theta, W, I0: W_I0_model(theta, W, I0, k), theta, phi_0_slice, p0=initial_guess_W)
    width, max_intensity = np.abs(popt_W_I0)
    perr_W_I0 = np.sqrt(np.diag(pcov_W_I0))
    W_err, max_intensity_error = perr_W_I0
    width_error = np.sqrt((W_err) ** 2 + (width * max_intensity_error) ** 2)

    popt_L, pcov_L = curve_fit(lambda phi, L: L_model(phi, L, max_intensity, k), phi, theta_0_slice, p0=initial_guess)
    height = popt_L[0]
    perr_L = np.sqrt(np.diag(pcov_L))
    height_error = np.sqrt((perr_L[0]) ** 2 + (height * max_intensity_error) ** 2)

    popt_D, pcov_D = curve_fit(lambda theta, D: D_model(theta, D, max_intensity, k), theta, theta_0_slice, p0=initial_guess)
    side_length = popt_D[0]
    perr_D = np.sqrt(np.diag(pcov_D))
    side_length_error = np.sqrt((perr_D[0]) ** 2 + (side_length * max_intensity_error) ** 2)
    sin_theta = np.sin(theta)
    popt_C, pcov_C = curve_fit(lambda theta, C: C_model(max_intensity, C, theta, np.sin(theta), k), diagonal_angle, diagonal_slice, p0=initial_guess)
    diameter = popt_C[0]
    perr_C = np.sqrt(np.diag(pcov_C))
    diameter_error = np.sqrt((perr_C[0]) ** 2 + (diameter * max_intensity_error) ** 2)

    popt_C2, pcov_C2 = curve_fit(lambda theta, C, k=k: C_model(max_intensity, C, theta, np.sin(theta), k), theta, theta_0_slice, p0=initial_guess)

    angular_diameter = popt_C2[0]

    shape, Area, Area_uncertainty, dimension1, dimension2, dimension1_err, dimension2_err = calculate_area_and_uncertainty(
        diameter, angular_diameter, width, height, side_length, diameter_error, width_error, height_error, side_length_error)

    power, power_uncertainty = calculate_power_and_uncertainty(max_intensity, Area, max_intensity_error,
                                                               Area_uncertainty)

    plot_diffraction_data(theta, phi, intensity)

    plot_intensity_slices(theta, diagonal_slice, diagonal_angle, theta_0_slice, phi, phi_0_slice, k)

main()


def ProcessData(filename):
    """Documentation string here."""
    #Your ProcessData code goes here

    #This is the data structure to return your results with -
    # replace the None values with your answers. Do not
    # rename any of the keys, do not delete any of the keys
    # - if your code doesn't find a value, leave it as None here.
    # Otherwise replace the None with the variable holding the correct answer from your code.
    results = {
        "shape": shape, # one of "square", "rectangle", "diamond", "circle" - must always be present.
        "dim_1": dimension1, # a floating point number of the first dimension expressed in microns
        "dim_1_err": dimension1_err, # The uncertainty in the above, also expressed in microns
        "dim_2": dimension2, # For a rectangle, the second dimension, for other shapes, the same as dim_1
        "dim_2_err": dimension2_err,  # The uncertainty in the above, also expressed in microns
        "I0/area": power, # The fitted overall intensity value/area of the aperture.
        "I0/area_err": power_uncertainty, # The uncertainty in the above.
    }
    
    return results

if __name__=="__main__":
     # Put your test code in side this if statement to stop 
     #it being run when you import your code

     filename="assessment_data_py22ct.dat"
     test_results=ProcessData(filename)
     print(test_results)