import { StorageType } from "../const/config"
import { ServerConfig } from "../types/config"

/* default config */
export const serverConfig: ServerConfig = {
  http: {
    port: 8686
  },
  storage: {
    type: StorageType.LOCAL,
    data: "./",
    itemDir: ""
  },
  user: {
    on: false
  },
  mode: {
    sync: true,
    autosave: true,
    dev: false,
    demo: false
  },
  redis: {
    writebackTime: 300,
    writebackCount: 32,
    port: 6379
  },
  bot: {
    on: false,
    host: "http://0.0.0.0",
    port: 8080
  }
}

/* default categories when file is missing and label is box2D or box3D */
const boxCategoriesList = [
  "pedestrian",
  "rider",
  "other person",
  "car",
  "bus",
  "truck",
  "train",
  "trailer",
  "other vehicle",
  "motorcycle",
  "bicycle",
  "traffic sign",
  "traffic light"
]
export const boxCategories = boxCategoriesList.map((category) => ({
  name: category
}))

/* default categories when file is missing and label is polyline2d */
export const polyline2DCategoriesList = [
  "road curb",
  "double white",
  "double yellow",
  "double other",
  "single white",
  "single yellow",
  "single other",
  "crosswalk"
]
export const polyline2DCategories = polyline2DCategoriesList.map(
  (category) => ({ name: category })
)

// TODO: add default seg2d categories once nested categories are supported

/* default attributes when file is missing and label is box2D */
export const box2DAttributes = [
  // {
  //   name: "Occluded",
  //   type: AttributeToolType.SWITCH,
  //   tag: "o",
  //   tagSuffixes: [],
  //   tagPrefix: "",
  //   values: [],
  //   buttonColors: []
  // },
  // {
  //   name: "Truncated",
  //   type: AttributeToolType.SWITCH,
  //   tag: "t",
  //   tagSuffixes: [],
  //   tagPrefix: "",
  //   values: [],
  //   buttonColors: []
  // },
  // {
  //   name: "Traffic Color Light",
  //   type: AttributeToolType.LIST,
  //   tag: "t",
  //   tagSuffixes: ["", "g", "y", "r"],
  //   tagPrefix: "",
  //   values: ["NA", "G", "Y", "R"],
  //   buttonColors: ["white", "green", "yellow", "red"]
  // }
]

/* default attributes when file is missing and no other defaults exist */
export const dummyAttributes = [
  // {
  //   name: "",
  //   type: AttributeToolType.NONE,
  //   tag: "",
  //   tagSuffixes: [],
  //   values: [],
  //   tagPrefix: "",
  //   buttonColors: []
  // }
]
