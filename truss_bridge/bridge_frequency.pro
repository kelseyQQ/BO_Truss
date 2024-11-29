init =
{
  nodeGroups = [ bl, br, bm, bottom ];

  mesh =
  {
    type = geo;
    file = bridge.geom;
  };

  bl = [0];
  br = [19];
  bm = [9];
  
  bottom = 
  {
    ytype = min;
  };
};

model =
{
  type = Multi;

  models = [ truss, mass, diri ];

  truss =
  {
    type = Truss;

    elements = all;

    subtype = linear;

    nsections = 15;
    young = 2.1e11;
    density = 7800;
    area = [4e-3,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4];

    shape =
    {
      type = Line2;
      intScheme = Gauss1;
    };
  };

  mass = 
  {
    type = PointMass;
    
    nodeGroup = bottom;
    mass = 10;
  };

  diri =
  {
    type = Dirichlet;

    groups = [ bl, bl, br ];
    dofs   = [ dx, dy, dy ];
    values = [ 0.0, 0.0, 0.0 ];
    dispIncr = [ 0.0, 0.0, 0.0 ];
  };
};

modeshape =
{
  type = ModeShape;
};

frameview =
{
  type = FrameView;
  deform = 1.;
};
