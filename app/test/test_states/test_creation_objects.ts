import _ from "lodash"

import {
  AttributeToolType,
  BundleFile,
  DataType,
  HandlerUrl,
  ItemTypeName,
  LabelTypeName
} from "../../src/const/common"
import { ItemExport } from "../../src/types/export"
import { CreationForm, FormFileData, Project } from "../../src/types/project"
import { Attribute, TaskType } from "../../src/types/state"
import {
  sampleStateExportImage,
  sampleStateExportImagePolygon
} from "./test_export_objects"

const sampleCategories: string[] = [
  "person",
  "rider",
  "car",
  "truck",
  "bus",
  "train",
  "motorcycle",
  "bike",
  "traffic sign",
  "traffic light"
]

const sampleAttributes: Array<Partial<Attribute>> = [
  {
    name: "Occluded",
    toolType: AttributeToolType.SWITCH,
    tagText: "o"
  },
  {
    name: "Truncated",
    toolType: AttributeToolType.SWITCH,
    tagText: "t"
  },
  {
    name: "Traffic Light Color",
    toolType: AttributeToolType.LIST,
    tagPrefix: "t",
    tagSuffixes: ["", "g", "y", "r"],
    values: ["NA", "G", "Y", "R"],
    buttonColors: ["white", "green", "yellow", "red"]
  }
]

export const sampleItems: Array<Partial<ItemExport>> = [
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg",
    videoName: "a",
    timestamp: 1
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg",
    videoName: "a",
    timestamp: 2
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg",
    videoName: "a",
    timestamp: 3
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg",
    videoName: "b",
    timestamp: 4
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg",
    videoName: "b",
    timestamp: 5
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000107.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000107.jpg",
    videoName: "b",
    timestamp: 6
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000108.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000108.jpg",
    videoName: "b",
    timestamp: 7
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000109.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000109.jpg",
    videoName: "b",
    timestamp: 8
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000110.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000110.jpg",
    videoName: "b",
    timestamp: 9
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000111.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000111.jpg",
    videoName: "b",
    timestamp: 10
  }
]

export const sampleFormEmpty: CreationForm = {
  projectName: "",
  itemType: "",
  labelType: "",
  pageTitle: "",
  taskSize: 0,
  instructionUrl: "",
  demoMode: false
}

const sampleProjectName = "sampleName"
const sampleInstructions = "instructions.com"
const sampleTitle = "sampleTitle"
const sampleTaskSize = 5

export const sampleFormImage: CreationForm = {
  projectName: sampleProjectName,
  itemType: ItemTypeName.IMAGE,
  labelType: LabelTypeName.BOX_2D,
  pageTitle: sampleTitle,
  taskSize: sampleTaskSize,
  instructionUrl: sampleInstructions,
  demoMode: false
}

export const sampleFormVideo: CreationForm = {
  projectName: sampleProjectName,
  itemType: ItemTypeName.VIDEO,
  labelType: LabelTypeName.POLYGON_2D,
  pageTitle: sampleTitle,
  taskSize: sampleTaskSize,
  instructionUrl: sampleInstructions,
  demoMode: true
}

export const sampleFormFileData: FormFileData = {
  categories: sampleCategories,
  attributes: sampleAttributes as Attribute[],
  sensors: [],
  items: sampleItems,
  templates: []
}

export const sampleProjectImage: Project = {
  items: sampleItems,
  config: {
    projectName: sampleProjectName,
    itemType: ItemTypeName.IMAGE,
    labelTypes: [LabelTypeName.BOX_2D],
    label2DTemplates: {},
    policyTypes: [],
    taskSize: sampleTaskSize,
    tracking: false,
    handlerUrl: HandlerUrl.LABEL,
    pageTitle: sampleTitle,
    instructionPage: sampleInstructions,
    bundleFile: BundleFile.V2,
    categories: sampleCategories,
    attributes: sampleAttributes as Attribute[],
    taskId: "",
    demoMode: false,
    autosave: true,
    bots: false
  },
  sensors: {}
}

const sampleItemsVideo = _.cloneDeep(sampleItems)
sampleItemsVideo[0].labels = [
  {
    id: "0",
    category: "person",
    attributes: {},
    manualShape: true,
    box2d: null,
    box3d: null,
    plane3d: null,
    poly2d: [
      {
        vertices: [
          [0, 0],
          [0, 1],
          [1, 1]
        ],
        types: "LLL",
        closed: true
      }
    ],
    customs: {}
  }
]
sampleItemsVideo[1].labels = [
  {
    id: "0",
    category: "person",
    attributes: {},
    manualShape: true,
    box2d: null,
    box3d: null,
    plane3d: null,
    poly2d: [
      {
        vertices: [
          [0, 0],
          [0, 1],
          [1, 1]
        ],
        types: "LLL",
        closed: true
      }
    ],
    customs: {}
  }
]

export const sampleVideoFormFileData: FormFileData = {
  categories: sampleCategories,
  attributes: sampleAttributes as Attribute[],
  items: sampleItemsVideo,
  sensors: [],
  templates: []
}

export const sampleProjectVideo: Project = {
  items: sampleItemsVideo,
  config: {
    projectName: sampleProjectName,
    itemType: ItemTypeName.IMAGE,
    labelTypes: [LabelTypeName.POLYGON_2D],
    label2DTemplates: {},
    policyTypes: [],
    taskSize: sampleTaskSize,
    tracking: true,
    handlerUrl: HandlerUrl.LABEL,
    pageTitle: sampleTitle,
    instructionPage: sampleInstructions,
    bundleFile: BundleFile.V1,
    categories: sampleCategories,
    attributes: sampleAttributes as Attribute[],
    taskId: "",
    demoMode: true,
    autosave: true,
    bots: false
  },
  sensors: {}
}

export const sampleProjectAutolabel: Project = {
  items: sampleStateExportImage,
  config: {
    projectName: sampleProjectName,
    itemType: ItemTypeName.IMAGE,
    labelTypes: [LabelTypeName.BOX_2D],
    label2DTemplates: {},
    policyTypes: [],
    taskSize: sampleTaskSize,
    tracking: false,
    handlerUrl: HandlerUrl.LABEL,
    pageTitle: sampleTitle,
    instructionPage: sampleInstructions,
    bundleFile: BundleFile.V2,
    categories: sampleCategories,
    attributes: sampleAttributes as Attribute[],
    taskId: "",
    demoMode: false,
    autosave: true,
    bots: false
  },
  sensors: {}
}

export const sampleProjectAutolabelPolygon: Project = {
  items: sampleStateExportImagePolygon,
  config: {
    projectName: sampleProjectName,
    itemType: ItemTypeName.IMAGE,
    labelTypes: [LabelTypeName.POLYGON_2D],
    label2DTemplates: {},
    policyTypes: [],
    taskSize: sampleTaskSize,
    tracking: false,
    handlerUrl: HandlerUrl.LABEL,
    pageTitle: sampleTitle,
    instructionPage: sampleInstructions,
    bundleFile: BundleFile.V2,
    categories: sampleCategories,
    attributes: sampleAttributes as Attribute[],
    taskId: "",
    demoMode: false,
    autosave: true,
    bots: false
  },
  sensors: {}
}

export const sampleTasksImage: TaskType[] = [
  {
    config: {
      projectName: sampleProjectName,
      itemType: ItemTypeName.IMAGE,
      labelTypes: [LabelTypeName.BOX_2D],
      label2DTemplates: {},
      policyTypes: [],
      taskSize: sampleTaskSize,
      tracking: false,
      handlerUrl: HandlerUrl.LABEL,
      pageTitle: sampleTitle,
      instructionPage: sampleInstructions,
      bundleFile: BundleFile.V2,
      categories: sampleCategories,
      attributes: sampleAttributes as Attribute[],
      taskId: "000000",
      demoMode: false,
      autosave: true,
      bots: false
    },
    status: {
      maxOrder: 0
    },
    progress: {
      submissions: []
    },
    items: [
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg"
        },
        index: 0,
        id: "0",
        labels: {},
        shapes: {},
        timestamp: 1,
        videoName: "a"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg"
        },
        index: 1,
        id: "1",
        labels: {},
        shapes: {},
        timestamp: 2,
        videoName: "a"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg"
        },
        index: 2,
        id: "2",
        labels: {},
        shapes: {},
        timestamp: 3,
        videoName: "a"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg"
        },
        index: 3,
        id: "3",
        labels: {},
        shapes: {},
        timestamp: 4,
        videoName: "b"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg"
        },
        index: 4,
        id: "4",
        labels: {},
        shapes: {},
        timestamp: 5,
        videoName: "b"
      }
    ],
    tracks: {},
    sensors: {
      [-1]: {
        id: -1,
        name: "default",
        type: DataType.IMAGE,
        extrinsics: undefined,
        intrinsics: undefined
      }
    }
  },
  {
    config: {
      projectName: sampleProjectName,
      itemType: ItemTypeName.IMAGE,
      labelTypes: [LabelTypeName.BOX_2D],
      label2DTemplates: {},
      policyTypes: [],
      taskSize: sampleTaskSize,
      tracking: false,
      handlerUrl: HandlerUrl.LABEL,
      pageTitle: sampleTitle,
      instructionPage: sampleInstructions,
      bundleFile: BundleFile.V2,
      categories: sampleCategories,
      attributes: sampleAttributes as Attribute[],
      taskId: "000001",
      demoMode: false,
      autosave: true,
      bots: false
    },
    status: {
      maxOrder: 0
    },
    progress: {
      submissions: []
    },
    items: [
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000107.jpg"
        },
        index: 0,
        id: "5",
        labels: {},
        shapes: {},
        timestamp: 6,
        videoName: "b"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000108.jpg"
        },
        index: 1,
        id: "6",
        labels: {},
        shapes: {},
        timestamp: 7,
        videoName: "b"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000109.jpg"
        },
        index: 2,
        id: "7",
        labels: {},
        shapes: {},
        timestamp: 8,
        videoName: "b"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000110.jpg"
        },
        index: 3,
        id: "8",
        labels: {},
        shapes: {},
        timestamp: 9,
        videoName: "b"
      },
      {
        urls: {
          [-1]: "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000111.jpg"
        },
        index: 4,
        id: "9",
        labels: {},
        shapes: {},
        timestamp: 10,
        videoName: "b"
      }
    ],
    tracks: {},
    sensors: {
      [-1]: {
        id: -1,
        name: "default",
        type: DataType.IMAGE,
        extrinsics: undefined,
        intrinsics: undefined
      }
    }
  }
]

export const sampleTasksVideo: TaskType[] = [
  {
    config: {
      projectName: "sampleName",
      itemType: "image",
      labelTypes: ["polygon2d"],
      label2DTemplates: {},
      policyTypes: [],
      taskSize: 3,
      tracking: true,
      handlerUrl: "label",
      pageTitle: "sampleTitle",
      instructionPage: "instructions.com",
      bundleFile: "image.js",
      categories: sampleCategories,
      attributes: sampleAttributes as Attribute[],
      taskId: "000000",
      demoMode: true,
      autosave: true,
      bots: false
    },
    status: {
      maxOrder: 2
    },
    progress: {
      submissions: []
    },
    items: [
      {
        id: "0",
        index: 0,
        videoName: "a",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg"
        },
        labels: {
          0: {
            id: "0",
            item: 0,
            sensors: [-1],
            type: "polygon2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["0"],
            track: "0",
            order: 0,
            manual: true
          }
        },
        shapes: {
          0: {
            id: "0",
            label: ["0"],
            shapeType: "polygon2d",
            points: [
              {
                x: 0,
                y: 0,
                pointType: "vertex"
              },
              {
                x: 0,
                y: 1,
                pointType: "vertex"
              },
              {
                x: 1,
                y: 1,
                pointType: "vertex"
              }
            ]
          }
        },
        timestamp: 1
      },
      {
        id: "1",
        index: 1,
        videoName: "a",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg"
        },
        labels: {
          1: {
            id: "1",
            item: 1,
            sensors: [-1],
            type: "polygon2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["1"],
            track: "0",
            order: 0,
            manual: true
          }
        },
        shapes: {
          1: {
            id: "1",
            label: ["1"],
            shapeType: "polygon2d",
            points: [
              {
                x: 0,
                y: 0,
                pointType: "vertex"
              },
              {
                x: 0,
                y: 1,
                pointType: "vertex"
              },
              {
                x: 1,
                y: 1,
                pointType: "vertex"
              }
            ]
          }
        },
        timestamp: 2
      },
      {
        id: "2",
        index: 2,
        videoName: "a",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 3
      }
    ],
    tracks: {
      0: {
        id: "0",
        type: "polygon2d",
        labels: {
          0: "0",
          1: "1"
        }
      }
    },
    sensors: {
      "-1": {
        id: -1,
        name: "default",
        type: "image",
        extrinsics: undefined,
        intrinsics: undefined
      }
    }
  },
  {
    config: {
      projectName: "sampleName",
      itemType: "image",
      labelTypes: ["polygon2d"],
      label2DTemplates: {},
      policyTypes: [],
      taskSize: 7,
      tracking: true,
      handlerUrl: "label",
      pageTitle: "sampleTitle",
      instructionPage: "instructions.com",
      bundleFile: "image.js",
      categories: sampleCategories,
      attributes: sampleAttributes as Attribute[],
      taskId: "000001",
      demoMode: true,
      autosave: true,
      bots: false
    },
    status: {
      maxOrder: 0
    },
    progress: {
      submissions: []
    },
    items: [
      {
        id: "3",
        index: 0,
        videoName: "b",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 4
      },
      {
        id: "4",
        index: 1,
        videoName: "b",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 5
      },
      {
        id: "5",
        index: 2,
        videoName: "b",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000107.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 6
      },
      {
        id: "6",
        index: 3,
        videoName: "b",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000108.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 7
      },
      {
        id: "7",
        index: 4,
        videoName: "b",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000109.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 8
      },
      {
        id: "8",
        index: 5,
        videoName: "b",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000110.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 9
      },
      {
        id: "9",
        index: 6,
        videoName: "b",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000111.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 10
      }
    ],
    tracks: {},
    sensors: {
      "-1": {
        id: -1,
        name: "default",
        type: "image",
        extrinsics: undefined,
        intrinsics: undefined
      }
    }
  }
]
