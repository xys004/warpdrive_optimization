(* ::Package:: *)

(* Render cleaned Python diagnostic-plane exports with Mathematica styling.

   This script is intentionally presentation-focused. The raw fields are
   exported by export_mathematica_plane_maps.py from the audited Python bundle,
   and this Wolfram script only handles figure aesthetics for manuscript use.
*)

args = If[
  ValueQ[$ScriptCommandLine] && Length[$ScriptCommandLine] >= 1,
  Rest[$ScriptCommandLine],
  {}
];

getArg[name_, default_] := Module[{pos = FirstPosition[args, name, Missing["NotFound"]]},
  If[pos === Missing["NotFound"] || pos[[1]] == Length[args], default, args[[pos[[1]] + 1]]]
];

inputDir = getArg["--input-dir", "."];
outputDir = getArg["--output-dir", FileNameJoin[{inputDir, "rendered"}]];
imageSize = ToExpression[getArg["--image-size", "420"]];
plotMin = ToExpression[getArg["--plot-min", "-2.6"]];
plotMax = ToExpression[getArg["--plot-max", "2.6"]];

If[!DirectoryQ[inputDir],
  Print["Input directory not found: ", inputDir];
  Exit[1];
];

If[!DirectoryQ[outputDir], CreateDirectory[outputDir, CreateIntermediateDirectories -> True]];

manifest = Import[FileNameJoin[{inputDir, "manifest.json"}], "RawJSON"];

softColorMap = Blend[
  {
    RGBColor[0.0, 0.0, 0.7],
    RGBColor[0.7, 0.7, 1.0],
    White,
    RGBColor[1.0, 0.7, 0.7],
    RGBColor[0.7, 0.0, 0.0]
  },
  #
] &;

fieldLabel[field_] := Lookup[
  <|
    "rho" -> "Energy density: rho",
    "WEC_min" -> "WEC min",
    "NEC_min" -> "NEC min",
    "DEC_margin" -> "DEC margin",
    "SEC" -> "SEC"
  |>,
  ToString[field],
  ToString[field]
];

planeLabel[plane_] := Lookup[
  <|
    "XY" -> "x-y",
    "XZ" -> "x-z"
  |>,
  ToString[plane],
  ToString[plane]
];

toPoints[json_] := Module[{x, y, values},
  x = json["axis_x"];
  y = json["axis_y"];
  values = json["values"];
  Cases[
    Flatten[
      Table[
        If[NumericQ[values[[i, j]]], {x[[i]], y[[j]], values[[i, j]]}, Nothing],
        {i, Length[x]}, {j, Length[y]}
      ],
      1
    ],
    {_?NumericQ, _?NumericQ, _?NumericQ}
  ]
];

renderField[field_, plane_, filename_] := Module[
  {json, points, clip, title, plot},
  json = Import[FileNameJoin[{inputDir, filename}], "RawJSON"];
  points = toPoints[json];
  clip = Lookup[json["display_limits"], "abs_q98", 1.0];
  If[!NumericQ[clip] || clip <= 0, clip = 1.0];
  title = fieldLabel[field] <> " (" <> planeLabel[plane] <> ")";

  plot = ListDensityPlot[
    points,
    PlotRange -> {{plotMin, plotMax}, {plotMin, plotMax}, {-clip, clip}},
    InterpolationOrder -> 2,
    ColorFunction -> (softColorMap[Rescale[#, {-clip, clip}]] &),
    ColorFunctionScaling -> False,
    PlotLegends -> None,
    Mesh -> None,
    Frame -> True,
    Axes -> False,
    FrameLabel -> {ToLowerCase[json["xlabel"]], ToLowerCase[json["ylabel"]]},
    PlotLabel -> Style[title, 10, GrayLevel[0.35]],
    BaseStyle -> {FontFamily -> "Times", FontSize -> 9},
    FrameStyle -> Directive[GrayLevel[0.78], Thickness[0.0015]],
    FrameTicksStyle -> Directive[GrayLevel[0.45], 7],
    Background -> White,
    ImageSize -> imageSize,
    PerformanceGoal -> "Quality"
  ];

  Export[
    FileNameJoin[{outputDir, StringReplace[filename, ".json" -> ".png"]}],
    plot,
    "PNG",
    ImageResolution -> 300
  ];
];

Do[
  With[{planeEntry = manifest["planes"][plane]},
    Do[
      renderField[field, plane, planeEntry["fields"][field]],
      {field, Keys[planeEntry["fields"]]}
    ]
  ],
  {plane, Keys[manifest["planes"]]}
];

Print["Rendered Mathematica figures to: ", outputDir];
