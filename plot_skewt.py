import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp

#defining constants needed for equations
epsilon = 0.622  # Ratio of the gas constants for dry air and water vapor
R_d = 287.05  # Gas constant for dry air (J/(kg·K))
R_v = 461.5  # Gas constant for water vapor (J/(kg·K))
c_pd = 1005.7  # Specific heat capacity of dry air at constant pressure (J/(kg·K))
L_v = 2.5e6  # Latent heat of vaporization (J/kg)
g = 9.81  # Gravitational acceleration (m/s^2)

#defining helper functions for skewt calculations
def e_s(temperature): 
    """ 
    Calculate the saturation vapor pressure over liquid water (in hPa) using the Bolton formula

    Parameters
    ----------
    temperature : array-like
        Temperature values in degrees Celsius.
    """
    return 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))

def r_s(temperature, pressure): 
    """ 
    Calculate the saturation mixing ratio (in kg/kg) using the saturation vapor pressure and pressure.

    Parameters
    ----------
    temperature : array-like
            Temperature values in degrees Celsius.
    pressure : array-like
            Pressure values in hPa.
    """
    return epsilon * e_s(temperature) / (pressure - e_s(temperature))

def moist_adiabatic_lapse_rate(pressure, temperature): #calculating the moist adiabatic lapse rate
    """ 
    Calculate the moist adiabatic lapse rate ()

    Parameters
    ----------
    temperature : array-like
        Temperature values in degrees Celsius.
    pressure : array-like
        Pressure values in hPa.
    """
    temp_k = temperature[0] + 273.15 
    
    r_s_value = r_s(temperature[0], pressure)
    
    numerator = 1 + (L_v * r_s_value) / (R_d * temp_k)
    denominator = 1 + (L_v**2 * r_s_value) / (c_pd * R_v * temp_k**2)
    dT_dz = (g / c_pd) * (numerator / denominator)
    
    dT_dP = (R_d * temp_k) / (pressure * g) * dT_dz #converting the results of the above equation to have pressure as a vertical coordinate
    
    return [dT_dP]

def skewT(temperature, pressure, skew=-20, ):
    """
    Skew the temperature axis for a Skew-T log-P plot.

    Parameters
    ----------
    temperature : array-like
        Temperature values in degrees Celsius.
    pressure : array-like
        Pressure values in hPa.
    skew : float, optional
        Amount of skewing, by default -20.
    """
    return temperature + skew * np.log(pressure / 1000.0)

def create_skewt(temperature, pressure, dewpoint):
    """
    Plot a Skew-T log-P diagram.
    
    Parameters
    ----------
    temperature : array-like
        Temperature values in degrees Celsius.
    pressure : array-like
        Pressure values in hPa.
    dewpoint : array-like
        Dewpoint temperature values in degrees Celsius, by default None.
        """
        
    fig, ax = plt.subplots(figsize=(10, 9)) #setting up figure to plot on
    
    #temperature profile
    skewed_temp = skewT(temperature, pressure, skew=-40) #Skew the temperature profile using observed temperature and pressure values
    ax.plot(skewed_temp, pressure, color='red', label='Temperature') # Plot temperature profile
    
        
    #dewpoint profile
    skewed_dewpoint = skewT(dewpoint, pressure, skew=-40) #Skew the dewpoint profile using observed dewpoint and pressure values
    ax.plot(skewed_dewpoint, pressure, color='green', label='Dewpoint') #Plot the dew point profile
    
     #isotherms
    pressure_range_isotherms = np.arange(1000, 0, -100) #specific pressure range for plotting isotherms since it does not require high resolution
    isotherms = np.arange(-80, 50, 10) #setting a range to plot ever 10°C
    for temp in isotherms: #looping through array of isotherms to skew each value and plot it on the figure
        skewed_isotherm = skewT(np.full_like(pressure_range_isotherms, temp), pressure_range_isotherms, skew=-40)
        ax.plot(skewed_isotherm, pressure_range_isotherms, color='gray', linewidth=0.5)
    
    #dry adiabats
    temp_range = np.arange(-40, 60, 10) #setting array for adiabats so one begins every 10°C
    pressure_range_dry = np.arange(1000, 90,-10) #specific vertical array for dry adiabats since it requires slightly more vertical resolution than isotherms
    for temp in temp_range: #looping through each value of temperature
        temp_k = temp + 273.15  #converting each temperature to Kelvin
        p0 = 1000 #initial surface pressure
        theta = temp_k * (pressure_range_dry / p0) ** (R_d/c_pd) #converting temperatures to potential temp (theta) using Poisson's equation
        skewed_theta_c = skewT(theta - 273.15, pressure_range_dry, skew=-40) #converting the temperature back into celsius and skewing
        ax.plot(skewed_theta_c, pressure_range_dry, color='red', linestyle='--', linewidth=1.0, label='Dry adiabats') #plotting the adiabat
    
    #moist adiabats
    added_temps = np.array([5, 15, 25, 35, 45]) #adding frequency of adiabats after 0°C
    moist_ad_temps = np.append(temp_range, added_temps) #using same temperatures as dry adiabats with added values
    pressure_range_moist = np.arange(1000, 90, -10) #specific pressure array since dry adiabats require greatest resolution in the vertical
    p_span = (1000, 100) #interval of integration
    for temp in moist_ad_temps: #looping through each temperature
        solve_moist_ad = solve_ivp(moist_adiabatic_lapse_rate, t_span=p_span, y0=[temp], t_eval=pressure_range_moist) #integrating the ODE in the vertical with each 
        #initial condition being the temperature. T_eval is pressure values in which data is stored so a greater resolution defined above ensures more data points (smoother curve)
        temp_c_profile = solve_moist_ad.y[0] #extracting the solution array
        
        skewed_moist_ad = skewT(temp_c_profile, pressure_range_moist, skew=-40) #skewing the solution with the pressure array
        ax.plot(skewed_moist_ad, pressure_range_moist, color='blue', linestyle='--', linewidth=1.0, label='Moist adiabats') #plotting moist adiabat
        
        
        
    ax.set_yscale('log') #making axis log-scale
    ax.invert_yaxis() #inverting the axis to put highest pressure at the bottom
    ax.yaxis.set_major_formatter('{x:.0f}') #formatting the pressure labels correctly
    ax.yaxis.set_minor_formatter('{x:.0f}')
    ax.grid(which='minor', axis='y', color='gray') #adding horizontal lines for each pressure level
    ax.set_ylim(1000, 100) #setting y-axis limits based on assignment instructions
    ax.set_ylabel('Pressure (hPa)') #adding label to y-axis
    
    ax.set_xlim(-40, 50) #setting y-axis limits based on assignment instructions
    ax.set_xlabel('Temperature (°C)') #adding label to x-axis
        
    ax.set_title('Skew-T log-P Diagram') # Set title
        
        
    plt.show()