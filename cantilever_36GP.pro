init =
{
  nodeGroups = [ left, top ];

  mesh =
  {
    type = geo;
    file = cantilever_36GP.geom;
  };

  top = 
  {
    ytype = max;
  };

  left = 
  {
    xtype = min;
  };
};

model =
{
  type = Multi;

  models = [ truss, load, diri ];

  truss =
  {
    type = Truss;

    elements = all;

    subtype = linear;

    nsections = 4;
    young = 1e4;
    density = 0.1e-3;
    area = [10,10,10,10];

    shape =
    {
      type = Line2;
      intScheme = Gauss1;
    };
  };

  load = 
  {
    type = Neumann;
    
    groups = [ top ];
    dofs = [ dy ];
    values = [ -20 ];
  };

  diri =
  {
    type = Dirichlet;

    groups = [ left, left ];
    dofs   = [ dx, dy ];
    values = [ 0.0, 0.0 ];
    dispIncr = [ 0.0, 0.0 ];
  };
};

solver =
{
  nsteps = 1;
};

frameview =
{
  type = FrameView;
  deform = 1.;
  interactive = False;
  plotStress = N;
};


