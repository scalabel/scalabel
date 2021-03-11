import { AttributeToolType } from "../../src/const/common"
import { DeepPartialState } from "../../src/types/state"

export const testJson: DeepPartialState = {
  session: {
    itemStatuses: [
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} }
    ],
    startTime: 1539820189
  },
  task: {
    config: {
      projectName: "Redux0",
      itemType: "image",
      labelTypes: ["box2d", "polygon2d", "tag", "polyline2d", "basicHumanPose"],
      label2DTemplates: {
        basicHumanPose: {
          name: "Basic Human Pose",
          nodes: [
            { name: "Head", x: 5, y: 5, color: [0, 0, 255] },
            { name: "Neck", x: 5, y: 10, color: [0, 0, 255] },
            { name: "Left Shoulder", x: 1, y: 10, color: [0, 255, 0] },
            { name: "Right Shoulder", x: 9, y: 10, color: [255, 0, 0] },
            { name: "Left Elbow", x: 1, y: 13, color: [0, 255, 0] },
            { name: "Right Elbow", x: 9, y: 13, color: [255, 0, 0] },
            { name: "Left Wrist", x: 1, y: 17, color: [0, 255, 0] },
            { name: "Right Wrist", x: 9, y: 17, color: [255, 0, 0] },
            { name: "Pelvis", x: 5, y: 22, color: [0, 0, 255] },
            { name: "Left Knee", x: 3, y: 30, color: [0, 255, 0] },
            { name: "Right Knee", x: 7, y: 30, color: [255, 0, 0] },
            { name: "Left Foot", x: 3, y: 40, color: [0, 255, 0] },
            { name: "Right Foot", x: 7, y: 40, color: [255, 0, 0] }
          ],
          edges: [
            [0, 1],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
            [1, 8],
            [8, 9],
            [8, 10],
            [9, 11],
            [10, 12]
          ]
        }
      },
      policyTypes: [
        "linear_interpolation_box_2d",
        "linear_interpolation_polygon"
      ],
      tracking: false,
      taskSize: 5,
      handlerUrl: "label2dv2",
      pageTitle: "Image Tagging Labeling Tool",
      instructionPage: "undefined",
      demoMode: false,
      autosave: false,
      categories: ["1", "2", "3"],
      attributes: [
        {
          name: "Weather",
          toolType: AttributeToolType.LIST,
          tagText: "",
          tagPrefix: "w",
          tagSuffixes: ["", "r", "s", "c", "o", "p", "f"],
          values: [
            "NA",
            "Rainy",
            "Snowy",
            "Clear",
            "Overcast",
            "Partly Cloudy",
            "Foggy"
          ],
          buttonColors: [
            "white",
            "white",
            "white",
            "white",
            "white",
            "white",
            "white"
          ]
        },
        {
          name: "Scene",
          toolType: AttributeToolType.LIST,
          tagText: "",
          tagPrefix: "s",
          tagSuffixes: ["", "t", "r", "p", "c", "g", "h"],
          values: [
            "NA",
            "Tunnel",
            "Residential",
            "Parking Lot",
            "City Street",
            "Gas Stations",
            "Highway"
          ],
          buttonColors: [
            "white",
            "white",
            "white",
            "white",
            "white",
            "white",
            "white"
          ]
        },
        {
          name: "Timeofday",
          toolType: AttributeToolType.LIST,
          tagText: "",
          tagPrefix: "t",
          tagSuffixes: ["", "day", "n", "daw"],
          values: ["NA", "Daytime", "Night", "Dawn/Dusk"],
          buttonColors: ["white", "white", "white", "white"]
        }
      ],
      taskId: "000000"
    },
    status: {
      maxOrder: -1
    },
    items: [
      {
        id: "id0",
        index: 0,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/" +
            "scalabel-public/demo/frames/intersection-0000051.jpg"
        },
        labels: {}
      },
      {
        id: "id1",
        index: 1,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/" +
            "demo/frames/intersection-0000052.jpg"
        },
        labels: {}
      },
      {
        id: "id2",
        index: 2,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/" +
            "demo/frames/intersection-0000053.jpg"
        },
        labels: {}
      },
      {
        id: "id3",
        index: 3,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/" +
            "demo/frames/intersection-0000054.jpg"
        },
        labels: {}
      },
      {
        id: "ABC_alphabeticallyOutOfOrderID",
        index: 4,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/" +
            "demo/frames/intersection-0000055.jpg"
        },
        labels: {}
      }
    ],
    tracks: {},
    sensors: {
      "-1": {
        id: -1,
        name: "default",
        type: "image"
      }
    },
    progress: {
      submissions: []
    }
  },
  user: {
    select: {
      item: 0,
      labelType: 0
    }
  }
}

const dummyViewerConfig = {
  imageWidth: 800,
  imageHeight: 800,
  viewScale: 1
}

/**
 * make a dummy item for testing
 *
 * @param id
 * @param labels
 * @param shapes
 */
function makeItem(id: number, labels: object, shapes: object): object {
  return {
    id,
    index: 0,
    loaded: false,
    labels,
    shapes,
    viewerConfig: dummyViewerConfig
  }
}

const dummyLabels = {
  0: {
    id: 0,
    category: [0],
    attributes: { 0: [0] },
    shapes: [0]
  },
  1: {
    id: 1,
    category: [1],
    attributes: { 1: [1] },
    shapes: [1]
  }
}

export const dummyNewLabel = {
  id: 1,
  item: 1,
  type: "bbox",
  category: [],
  attributes: {},
  parent: 1,
  children: [],
  shapes: [2],
  state: 2,
  order: 2
}

const dummyShapes = {
  0: {
    id: 0,
    label: [0],
    shape: {
      x1: 0,
      y1: 0,
      x2: 1,
      y2: 1
    }
  },
  1: {
    id: 1,
    label: [1],
    shape: { x1: 2, y1: 2, x2: 3, y2: 3 }
  }
}

export const autoSaveTestJson = {
  items: [makeItem(0, dummyLabels, dummyShapes), makeItem(1, {}, {})],
  current: {
    item: 0,
    maxLabelId: 1,
    maxShapeId: 1,
    maxOrder: 1
  }
}
