import itertools
import os

# Define all binary parameter names
parameter_names = ["x_12", "x_13", "x_14", "x_23", "x_24", "x_34"]

# Generate all possible combinations of binary values (2^6 = 64 combinations)
combinations = list(itertools.product([0, 1], repeat=len(parameter_names)))

# System properties
# Environmental Parameters
T_env = 293       # Ambient temperature (K)
R_env = 0.01      # Convection resistance to ambient (K/W)

# Heat Flows (in Watts)
# Positive values indicate heat generation; negative values indicate cooling.
Q_1 = 2000    
Q_2 = 4000    
Q_3 = -200    
Q_4 = -2000   

# Inter-node Thermal Resistances (in K/W)
# These values lump together conduction and convection effects.
R_12 = 0.005
R_13 = 0.006
R_14 = 0.006
R_23 = 0.007
R_24 = 0.007
R_34 = 0.008

# Base model structure
def base_model_template(x_12, x_13, x_14, x_23, x_24, x_34): 
  template = f"""
model cooling_system

  // Define binary parameters (true = connection exists, false = no connection)
  parameter Boolean x_12 = {x_12};
  parameter Boolean x_13 = {x_13};
  parameter Boolean x_14 = {x_14};
  parameter Boolean x_23 = {x_23};
  parameter Boolean x_24 = {x_24};
  parameter Boolean x_34 = {x_34};

  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_23(R = {R_23}) if x_23 annotation(
    Placement(transformation(origin = {{-46, -4}}, extent = {{{{-6, -6}}, {{6, 6}}}}, rotation = 90)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_12(R = {R_12}) if x_12 annotation(
    Placement(transformation(origin = {{20, 24}}, extent = {{{{-6, -6}}, {{6, 6}}}}, rotation = -90)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_24(R = {R_24}) if x_24 annotation(
    Placement(transformation(origin = {{30, -4}}, extent = {{{{-6, -6}}, {{6, 6}}}}, rotation = -90)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_34(R = {R_34}) if x_34 annotation(
    Placement(transformation(origin = {{0, -48}}, extent = {{{{-6, -6}}, {{6, 6}}}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_13(R = {R_13}) if x_13 annotation(
    Placement(transformation(origin = {{-14, -26}}, extent = {{{{-6, -6}}, {{6, 6}}}}, rotation = 270)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_14(R = {R_14}) if x_14 annotation(
    Placement(transformation(origin = {{14, -26}}, extent = {{{{-6, -6}}, {{6, 6}}}}, rotation = -90)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Cooler_3(C = 0.01) annotation(
    Placement(transformation(origin = {{-30, -38}}, extent = {{{{-10, -10}}, {{10, 10}}}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Engine_2(C = 0.01) annotation(
    Placement(transformation(origin = {{20, 60}}, extent = {{{{-10, -10}}, {{10, 10}}}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Cooler_4(C = 0.01) annotation(
    Placement(transformation(origin = {{30, -38}}, extent = {{{{-10, -10}}, {{10, 10}}}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Battery_1(C = 0.01) annotation(
    Placement(transformation(origin = {{0, 26}}, extent = {{{{-10, -10}}, {{10, 10}}}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow Q_3(Q_flow = {Q_3}) annotation(
    Placement(transformation(origin = {{-30, -64}}, extent = {{{{-10, -10}}, {{10, 10}}}}, rotation = 90)));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow Q_2(Q_flow = {Q_2}) annotation(
    Placement(transformation(origin = {{2, 50}}, extent = {{{{-10, -10}}, {{10, 10}}}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow Q_1(Q_flow = {Q_1}) annotation(
    Placement(transformation(origin = {{-18, 16}}, extent = {{{{-10, -10}}, {{10, 10}}}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow Q_4(Q_flow = {Q_4}) annotation(
    Placement(transformation(origin = {{30, -64}}, extent = {{{{10, 10}}, {{-10, -10}}}}, rotation = -90)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_env_4(R = {R_env}) annotation(
    Placement(transformation(origin = {{44, -48}}, extent = {{{{-4, -4}}, {{4, 4}}}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_env_1(R = {R_env}) annotation(
    Placement(transformation(origin = {{-12, -4}}, extent = {{{{-4, -4}}, {{4, 4}}}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_env_3(R = {R_env}) annotation(
    Placement(transformation(origin = {{-44, -48}}, extent = {{{{-4, -4}}, {{4, 4}}}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_env_2(R = {R_env}) annotation(
    Placement(transformation(origin = {{34, 50}}, extent = {{{{-4, -4}}, {{4, 4}}}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature Environment_4(T (displayUnit = "K")= {T_env}) annotation(
    Placement(transformation(origin = {{58, -48}}, extent = {{{{8, -8}}, {{-8, 8}}}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature Environment_3(T (displayUnit = "K")= {T_env}) annotation(
    Placement(transformation(origin = {{-58, -48}}, extent = {{{{-8, -8}}, {{8, 8}}}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature Environment_1(T (displayUnit = "K")= {T_env}) annotation(
    Placement(transformation(origin = {{-26, -4}}, extent = {{{{-8, -8}}, {{8, 8}}}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature Environment_2(T (displayUnit = "K")= {T_env}) annotation(
    Placement(transformation(origin = {{48, 50}}, extent = {{{{8, -8}}, {{-8, 8}}}}, rotation = -0)));
equation
  // Conditional connections
  if x_12 then
    connect(Engine_2.port, R_12.port_a) annotation(
      Line(points = {{{{20, 50}}, {{20, 30}}}}, color = {{191, 0, 0}}));
    connect(R_12.port_b, Battery_1.port) annotation(
      Line(points = {{{{20, 18}}, {{20, 16}}, {{0, 16}}}}, color = {{191, 0, 0}}));
      
  end if;

  if x_13 then
    connect(Cooler_3.port, R_13.port_b) annotation(
      Line(points = {{{{-30, -48}}, {{-14, -48}}, {{-14, -32}}}}, color = {{191, 0, 0}}));
    connect(R_13.port_a, Battery_1.port) annotation(
      Line(points = {{{{-14, -20}}, {{-2, -20}}, {{-2, 15}}, {{0, 15}}, {{0, 16}}}}, color = {{191, 0, 0}}));
  end if;

  if x_14 then
    connect(Cooler_4.port, R_14.port_b) annotation(
      Line(points = {{{{30, -48}}, {{14, -48}}, {{14, -32}}}}, color = {{191, 0, 0}}));
    connect(R_14.port_a, Battery_1.port) annotation(
      Line(points = {{{{14, -20}}, {{2, -20}}, {{2, 16}}, {{0, 16}}}}, color = {{191, 0, 0}}));
  end if;
  if x_23 then
    connect(Cooler_3.port, R_23.port_a) annotation(
      Line(points = {{{{-30, -48}}, {{-38, -48}}, {{-38, -24}}, {{-46, -24}}, {{-46, -10}}}}, color = {{191, 0, 0}}));
    connect(R_23.port_b, Engine_2.port) annotation(
      Line(points = {{{{-46, 2}}, {{-46, 40}}, {{20, 40}}, {{20, 50}}}}, color = {{191, 0, 0}}));
  end if;
  if x_24 then
    connect(Cooler_4.port, R_24.port_b) annotation(
      Line(points = {{{{30, -48}}, {{22, -48}}, {{22, -24}}, {{30, -24}}, {{30, -10}}}}, color = {{191, 0, 0}}));
    connect(R_24.port_a, Engine_2.port) annotation(
      Line(points = {{{{30, 2}}, {{30, 42}}, {{20, 42}}, {{20, 50}}}}, color = {{191, 0, 0}}));
  end if;
  if x_34 then
    connect(Cooler_3.port, R_34.port_a) annotation(
      Line(points = {{{{-30, -48}}, {{-6, -48}}}}, color = {{191, 0, 0}}));
    connect(R_34.port_b, Cooler_4.port) annotation(
      Line(points = {{{{6, -48}}, {{30, -48}}}}, color = {{191, 0, 0}}));
  end if;
  connect(Engine_2.port, Q_2.port) annotation(
    Line(points = {{{{20, 50}}, {{12, 50}}}}, color = {{191, 0, 0}}));
  connect(R_env_3.port_b, Cooler_3.port) annotation(
    Line(points = {{{{-40, -48}}, {{-30, -48}}}}, color = {{191, 0, 0}}));
  connect(Cooler_3.port, Q_3.port) annotation(
    Line(points = {{{{-30, -48}}, {{-30, -54}}}}, color = {{191, 0, 0}}));
  connect(R_env_3.port_a, Environment_3.port) annotation(
    Line(points = {{{{-48, -48}}, {{-50, -48}}}}, color = {{191, 0, 0}}));
  connect(Cooler_4.port, Q_4.port) annotation(
    Line(points = {{{{30, -48}}, {{30, -54}}}}, color = {{191, 0, 0}}));
  connect(Cooler_4.port, R_env_4.port_a) annotation(
    Line(points = {{{{30, -48}}, {{40, -48}}}}, color = {{191, 0, 0}}));
  connect(R_env_4.port_b, Environment_4.port) annotation(
    Line(points = {{{{48, -48}}, {{50, -48}}}}, color = {{191, 0, 0}}));
  connect(Environment_1.port, R_env_1.port_a) annotation(
    Line(points = {{{{-18, -4}}, {{-16, -4}}}}, color = {{191, 0, 0}}));
  connect(R_env_1.port_b, Battery_1.port) annotation(
    Line(points = {{{{-8, -4}}, {{-4, -4}}, {{-4, 16}}, {{0, 16}}, {{0, 16}}}}, color = {{191, 0, 0}}));
  connect(Engine_2.port, R_env_2.port_a) annotation(
    Line(points = {{{{20, 50}}, {{30, 50}}}}, color = {{191, 0, 0}}));
  connect(R_env_2.port_b, Environment_2.port) annotation(
    Line(points = {{{{38, 50}}, {{40, 50}}}}, color = {{191, 0, 0}}));
  connect(Q_1.port, Battery_1.port) annotation(
    Line(points = {{{{-8, 16}}, {{0, 16}}}}, color = {{191, 0, 0}}));
  annotation(
    uses(Modelica(version = "4.0.0")),
  Diagram(coordinateSystem(extent = {{{{-80, 80}}, {{80, -80}}}})),
  version = "");
end cooling_system;
  """

  return template

# Output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'cooling_system_models')

# Generate each model
for combination in combinations:
    # Generate model name
    model_name = "cooling_system_" + "".join(map(str, combination))
    
    # Create parameter assignment dictionary
    parameters = ["true" if bit==1 else "false" for bit in combination]
    # Create the model definition
    model_definition = base_model_template(*parameters)
    
    # Write the model to its own file
    file_name = f"{output_dir}/{model_name}.mo"
    with open(file_name, "w") as f:
        f.write(model_definition)
    
print(f"Generated {len(combinations)} stand-alone Modelica files in '{output_dir}'")
