import "ol/ol.css";
import { Map, View, Feature } from "ol";
import TileLayer from "ol/layer/Tile";
import OSM from "ol/source/OSM";
import { fromLonLat } from "ol/proj";
import { Point } from "ol/geom";
import { Vector as VectorSource } from "ol/source";
import { Vector as VectorLayer } from "ol/layer";
import { Circle as CircleStyle, Fill, Stroke, Style } from "ol/style.js";
import * as d3 from "d3";
import proj4 from "proj4";
import { pointerMove } from "ol/events/condition.js";
import Select from "ol/interaction/Select.js";

const MapOrigin = [-0.57918, 44.837789]; // Long Lat Bordeaux
const map = new Map({
  target: "map",
  layers: [
    new TileLayer({
      source: new OSM(),
      preload: Infinity
    })
  ],
  view: new View({
    center: fromLonLat(MapOrigin),
    zoom: 8
  })
});

// Convert Coordinates
const source =
  "+proj=lcc +lat_1=49 +lat_2=44 +lat_0=46.5 +lon_0=3 +x_0=700000 +y_0=6600000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ";
const dest = "WGS84";

var csvURL = "http://localhost:5000/data/etablissements_basique.csv";
d3.csv(csvURL).then(function(data) {
  var features = new Array(data.length);
  var scores = new Array(data.length);
  var maxScore = 0;
  var styles = new Array(data.length);

  for (var row = 0; row < data.length; row++) {
    var x = Number(data[row]["Coordonnee X"]);
    var y = Number(data[row]["Coordonnee Y"]);
    var score = Number(data[row]["score"]);
    scores[row] = score;
    maxScore = Math.max(maxScore, score);

    var lat_long = proj4(source, dest, [x, y]);
    features[row] = new Feature({
      geometry: new Point(fromLonLat(lat_long))
      /* i: row,
       * size: row % 2 ? 10 : 20 */
    });
  }

  for (var row = 0; row < scores.length; row++) {
    score = scores[row];
    function styleFunction(score) {
      var radius = (score * 15) / maxScore;
      console.log(radius);
      var style = new Style({
        image: new CircleStyle({
          radius: radius,
          fill: new Fill({
            color: "rgba(255, 153, 0, 0.4)"
          }),
          stroke: new Stroke({
            color: "rgba(255, 204, 0, 0.2)",
            width: 1
          })
        })
      });
      return style;
    }
    styles[row] = styleFunction(score);
  }

  var vectorSource = new VectorSource({
    features: features,
    wrapX: false
  });
  var vector = new VectorLayer({
    source: vectorSource,
    style: styles
  });
  map.addLayer(vector);
});

// Mouse hover
var HoverStyle = new Style({
  image: new CircleStyle({
    radius: 12,
    fill: new Fill({ color: "#666666" }),
    stroke: new Stroke({ color: "#bada55", width: 2 })
  })
});
// select interaction working on "pointermove"
var selectPointerMove = new Select({
  condition: pointerMove,
  style: HoverStyle
});
map.addInteraction(selectPointerMove);
