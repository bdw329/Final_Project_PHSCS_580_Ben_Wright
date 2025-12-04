# Link to article on Newtonian Hypersonic Method
# https://aerospaceweb.org/design/waverider/theory.shtml (basic theory)
# https://ntrs.nasa.gov/api/citations/20110013613/downloads/20110013613.pdf (Orion capsule data, may be better than using CFD)
# Chapter 3.3 of Anderson's hypersonic text (modified Newtonian theory)
import numpy as np
import matplotlib.pyplot as plt
from capsule_geometry import build_orion_capsule
from data import α_experimental, L_D_ratio_experimental
# Extract body points of orion
xbody, ybody = build_orion_capsule(a=1.0, total_points=300)
plt.plot(xbody,ybody)
plt.axis('equal')
plt.show()
#xbody = np.array([0, 1, 2, 1])
#ybody = np.array([0, 0.176, 0, -0.176])
Npanels = len(xbody)
# Function to calculate L/D using Modified Newtonian Law
def modified_newtonian_method(xbody, ybody, α, Minf):
    γ = 1.4
    α = (α + 180)*np.pi/180
    ρinf = 0.066 # slant density at 80km altitude (kg/m^3)
    pinf = 101325  # slant pressure at sea level (Pa)
    R_gas = 287  # specific gas constant for air (J/kg-K)
    Tinf = 61
    a = np.sqrt(γ * R_gas * Tinf)  # speed of sound (m/s)
    Vinf = Minf * a  # freestream velocity (m/s)

    L = 0
    D = 0

    xpoints = xbody.copy()
    ypoints = ybody.copy()
    xpoints_rot = xpoints*np.cos(α) - ypoints*np.sin(α)
    ypoints_rot = xpoints*np.sin(α) + ypoints*np.cos(α)
    xpoints = xpoints_rot
    ypoints = ypoints_rot

    # For loop to analyze panels
    for i in range(Npanels):
        j = (i + 1) % Npanels
        dy = ypoints[j] - ypoints[i]
        dx = xpoints[j] - xpoints[i]
        α_panel = -np.arctan2(-dy,dx)
        term1 = ((γ + 1)**2*Minf**2/(4*γ*Minf**2 - 2*(γ - 1)))**(γ/(γ - 1))
        term2 = (1 - γ + 2*γ*Minf**2)/(γ + 1)
        Cp_max = (2/(γ*Minf**2))*(term1*term2 - 1)
        Cp = Cp_max * np.sin(α_panel)**2
        Panel_length = np.sqrt(dx**2 + dy**2)
        Ftotal = Cp  * (0.5* ρinf * Vinf**2) * Panel_length

        Fx = Ftotal*np.sin(α_panel)
        Fy = Ftotal*np.cos(α_panel)
        if Fx < 0:
            L += 0
            D += 0
        else:
            L += Fy
            D += Fx

    Lift_to_Drag = L/D
    return Lift_to_Drag

α_range = np.linspace(-5, 170, 200)
L_D_Newtonian = np.zeros(len(α_range))
Lift_range = np.zeros(len(α_range))
Drag_range = np.zeros(len(α_range)) 
# Test method over a range of angles of attack
for i in range(len(α_range)):
    L_D_Newtonian[i] = modified_newtonian_method(xbody, ybody, α_range[i], 6)

plt.plot(α_range, L_D_Newtonian,
         label="Predicted L/D Using Modified Newtonian Theory")
plt.scatter(α_experimental, L_D_ratio_experimental,
            label="Experimental NASA data", color = "red")
plt.title("Lift to Drag Ratio vs. Angle of Attack at Mach 6")
plt.xlabel("α (degrees)")
plt.ylabel("Lift-to-Drag Ratio")
plt.legend()
# plt.grid(True, alpha=0.3)   # optional but looks good
plt.show()

def residual(α_experimental, L_D_ratio_experimental):
    r = np.zeros(len(α_experimental))
    for i in range(len(α_experimental)):
        r[i] = L_D_ratio_experimental[i] - modified_newtonian_method(xbody, ybody, α_experimental[i], 6)
    return r
# Calculate difference between experimental results and Modified Newtonian Results
L_D_ratio_residuals = residual(α_experimental, L_D_ratio_experimental)
plt.scatter(α_experimental, L_D_ratio_residuals)
plt.title("residual of predicted vs. experimental L_D_ratio results")
plt.xlabel("α (degrees)")
plt.ylabel("L/D ratrio residuals")
plt.show()

# Now create residual models. First, test a linear regression using only the experimental data.
def create_best_fit_model(α_range, order):

    A_experimental = np.zeros((len(α_experimental), order + 1))
    for i in range(len(α_experimental)):
        for j in range(order + 1):
            A_experimental[i,j] = α_experimental[i]**j

    theta_best_fit = np.linalg.inv(A_experimental.T @ A_experimental) @ (A_experimental.T @ L_D_ratio_experimental)

    L_D_best_fit_model = np.zeros(len(α_range))
    for i in range(len(α_range)):
        for j in range(len(theta_best_fit)):
            L_D_best_fit_model[i] += theta_best_fit[j]*α_range[i]**j 

    return L_D_best_fit_model

L_D_best_fit_model = create_best_fit_model(α_range, 6)
plt.plot(α_range, L_D_best_fit_model, label = "Predicted L/D model, using 4th order polynomial")
plt.scatter(α_experimental, L_D_ratio_experimental,
            label="Experimental NASA data", color = "red")
plt.legend()
plt.show()

# Create design matrix for residual function
def create_residual_model(α_range, order):

    L_D_Newtonian = np.zeros(len(α_experimental))
    for i in range(len(α_experimental)):
        L_D_Newtonian[i] = modified_newtonian_method(xbody, ybody, α_experimental[i], 6)

    L_D_ratio_residuals = L_D_ratio_experimental - L_D_Newtonian

    A_residuals = np.zeros((len(α_experimental), order + 1))
    for i in range(len(α_experimental)):
        for j in range(order + 1):
            A_residuals[i,j] = α_experimental[i]**j

    theta_residuals = np.linalg.inv(A_residuals.T @ A_residuals) @ (A_residuals.T @ L_D_ratio_residuals)

    # Now we have thetas

    L_D_ratio_residual_model = np.zeros(len(α_range))

    for i in range(len(α_range)):
        for j in range(len(theta_residuals)):
            L_D_ratio_residual_model[i] += theta_residuals[j]*α_range[i]**j

    L_D_Newtonian = np.zeros(len(α_range))
    for i in range(len(α_range)):
        L_D_Newtonian[i] = modified_newtonian_method(xbody, ybody, α_range[i], 6)
    
    L_D_residual_correction_model = L_D_ratio_residual_model + L_D_Newtonian

    return L_D_residual_correction_model

# Now plot experimental results, newtonian model, experimental fit model, residual model_D_residual_correction 
L_D_residual_correction_model = create_residual_model(α_range, 6)

plt.scatter(α_experimental, L_D_ratio_experimental, label = "Experimental Data", color = "red")
plt.plot(α_range, L_D_Newtonian, label = "Modified Newtonian Model")
plt.plot(α_range, L_D_best_fit_model, label = "Experimental Data Best Fit")
plt.plot(α_range, L_D_residual_correction_model, label = "Residual Correction Model")
plt.xlabel("Angle of attack (degrees)")
plt.ylabel("Lift to Drag Ratio")
plt.title("Plot Comparing Experimental Data to 3 Models (6th Order)")
plt.legend(loc="best")
plt.show()

L_D_Newtonian_l2norm = np.zeros(11)
L_D_best_fit_norm = np.zeros(11)
L_D_residual_norm = np.zeros(11)

# This creates a model of a desired order, and calculates the L2 Norm of the model against the experimental data
for order in range(11):
    L_D_Newtonian = np.zeros(len(α_experimental))
    for i in range(len(α_experimental)):
        L_D_Newtonian[i] = modified_newtonian_method(xbody, ybody, α_experimental[i], 6)
    L_D_Newtonian_residual = L_D_ratio_experimental - L_D_Newtonian
    L_D_Newtonian_l2norm[order] = np.linalg.norm(L_D_Newtonian_residual)

    L_D_best_fit_model = create_best_fit_model(α_experimental, order)
    L_D_best_fit_residual = L_D_ratio_experimental - L_D_best_fit_model
    L_D_best_fit_norm[order] = np.linalg.norm(L_D_best_fit_residual)

    L_D_residual_model = create_residual_model(α_experimental, order)
    L_D_residual_residual = L_D_ratio_experimental - L_D_residual_model
    L_D_residual_norm[order] = np.linalg.norm(L_D_residual_residual)

# Create plot comparing L2 Norms of different methods
order = np.linspace(0,10,11)
plt.plot(order, L_D_Newtonian_l2norm, label = "Newtonian Model L2 Norm")
plt.plot(order, L_D_best_fit_norm, label = "Best Fit L2 Norm")
plt.plot(order, L_D_residual_norm, label = "Residual Model Norm")
plt.xlabel("Order of Model")
plt.ylabel("L2 Norm of Lift to Drag Ratio")
plt.legend()
plt.title("Comparison of L2 Norms of Different Models")
plt.show()

