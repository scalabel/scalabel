import { AttributeToolType } from "../../src/const/common"
import { DeepPartialState } from "../../src/types/state"

export const testJson: DeepPartialState = {
  task: {
    config: {
      projectName: "Test3",
      itemType: "image",
      labelTypes: ["box2d"],
      label2DTemplates: {},
      taskSize: 6,
      handlerUrl: "label",
      pageTitle: "2D Bounding Box",
      instructionPage: "https://doc.scalabel.ai/instructions/bbox.html",
      bundleFile: "image_v2.js",
      categories: [
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
      ],
      attributes: [
        {
          name: "Occluded",
          toolType: AttributeToolType.SWITCH,
          tagText: "o",
          tagSuffixes: [],
          tagPrefix: "",
          values: [],
          buttonColors: []
        },
        {
          name: "Truncated",
          toolType: AttributeToolType.SWITCH,
          tagText: "t",
          tagSuffixes: [],
          tagPrefix: "",
          values: [],
          buttonColors: []
        },
        {
          name: "Traffic Light Color",
          toolType: AttributeToolType.LIST,
          tagText: "t",
          tagSuffixes: ["", "g", "y", "r"],
          tagPrefix: "",
          values: ["NA", "G", "Y", "R"],
          buttonColors: ["white", "green", "yellow", "red"]
        }
      ],
      taskId: "000000",
      tracking: true,
      policyTypes: ["linear_interpolation"],
      demoMode: false,
      autosave: true
    },
    status: {
      maxOrder: 223
    },
    items: [
      {
        id: "0",
        index: 0,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg"
        },
        labels: {
          id23: {
            id: "id23",
            item: 0,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["id23"],
            track: "1",
            order: 24,
            manual: true
          },
          id46: {
            id: "id46",
            item: 0,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["id46"],
            track: "2",
            order: 47,
            manual: true
          },
          id69: {
            id: "id69",
            item: 0,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["id69"],
            track: "3",
            order: 70,
            manual: true
          }
        },
        shapes: {
          id23: {
            id: "id23",
            label: ["id23"],
            shapeType: "rect",
            x1: 0,
            y1: 298.1679389312977,
            x2: 145.3482824427481,
            y2: 651.2977099236641
          },
          id46: {
            id: "id46",
            label: ["id46"],
            shapeType: "rect",
            x1: 950.5391221374047,
            y1: 287.17557251908397,
            x2: 1021.9895038167939,
            y2: 478.1679389312977
          },
          id69: {
            id: "id69",
            label: ["id69"],
            shapeType: "rect",
            x1: 835.1192748091603,
            y1: 314.65648854961836,
            x2: 861.2261450381679,
            y2: 406.71755725190843
          }
        },
        timestamp: 0
      },
      {
        id: "1",
        index: 1,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg"
        },
        labels: {
          24: {
            id: "24",
            item: 1,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["24"],
            track: "1",
            order: 25,
            manual: false
          },
          47: {
            id: "47",
            item: 1,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["47"],
            track: "2",
            order: 48,
            manual: false
          },
          70: {
            id: "70",
            item: 1,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["70"],
            track: "3",
            order: 71,
            manual: false
          }
        },
        shapes: {
          24: {
            id: "24",
            label: ["24"],
            shapeType: "rect",
            x1: 0,
            y1: 298.1679389312977,
            x2: 145.3482824427481,
            y2: 651.2977099236641
          },
          47: {
            id: "47",
            label: ["47"],
            shapeType: "rect",
            x1: 942.2948473282444,
            y1: 323.9312977099237,
            x2: 1013.7452290076336,
            y2: 514.9236641221374
          },
          70: {
            id: "70",
            label: ["70"],
            shapeType: "rect",
            x1: 835.1192748091603,
            y1: 314.65648854961836,
            x2: 861.2261450381679,
            y2: 406.71755725190843
          }
        },
        timestamp: 0
      },
      {
        id: "2",
        index: 2,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg"
        },
        labels: {
          25: {
            id: "25",
            item: 2,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["25"],
            track: "1",
            order: 26,
            manual: false
          },
          48: {
            id: "48",
            item: 2,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["48"],
            track: "2",
            order: 49,
            manual: false
          },
          71: {
            id: "71",
            item: 2,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["71"],
            track: "3",
            order: 72,
            manual: false
          }
        },
        shapes: {
          25: {
            id: "25",
            label: ["25"],
            shapeType: "rect",
            x1: 0,
            y1: 298.1679389312977,
            x2: 145.3482824427481,
            y2: 651.2977099236641
          },
          48: {
            id: "48",
            label: ["48"],
            shapeType: "rect",
            x1: 934.0505725190841,
            y1: 360.68702290076334,
            x2: 1005.5009541984733,
            y2: 551.679389312977
          },
          71: {
            id: "71",
            label: ["71"],
            shapeType: "rect",
            x1: 835.1192748091603,
            y1: 314.65648854961836,
            x2: 861.2261450381679,
            y2: 406.71755725190843
          }
        },
        timestamp: 0
      },
      {
        id: "3",
        index: 3,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg"
        },
        labels: {
          26: {
            id: "26",
            item: 3,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["26"],
            track: "1",
            order: 27,
            manual: false
          },
          49: {
            id: "49",
            item: 3,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["49"],
            track: "2",
            order: 50,
            manual: false
          },
          72: {
            id: "72",
            item: 3,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["72"],
            track: "3",
            order: 73,
            manual: false
          }
        },
        shapes: {
          26: {
            id: "26",
            label: ["26"],
            shapeType: "rect",
            x1: 0,
            y1: 298.1679389312977,
            x2: 145.3482824427481,
            y2: 651.2977099236641
          },
          49: {
            id: "49",
            label: ["49"],
            shapeType: "rect",
            x1: 925.8062977099238,
            y1: 397.4427480916031,
            x2: 997.256679389313,
            y2: 588.4351145038167
          },
          72: {
            id: "72",
            label: ["72"],
            shapeType: "rect",
            x1: 835.1192748091603,
            y1: 314.65648854961836,
            x2: 861.2261450381679,
            y2: 406.71755725190843
          }
        },
        timestamp: 0
      },
      {
        id: "4",
        index: 4,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg"
        },
        labels: {
          27: {
            id: "27",
            item: 4,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["27"],
            track: "1",
            order: 28,
            manual: false
          },
          73: {
            id: "73",
            item: 4,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["73"],
            track: "3",
            order: 74,
            manual: false
          }
        },
        shapes: {
          27: {
            id: "27",
            label: ["27"],
            shapeType: "rect",
            x1: 0,
            y1: 298.1679389312977,
            x2: 145.3482824427481,
            y2: 651.2977099236641
          },
          73: {
            id: "73",
            label: ["73"],
            shapeType: "rect",
            x1: 835.1192748091603,
            y1: 314.65648854961836,
            x2: 861.2261450381679,
            y2: 406.71755725190843
          }
        },
        timestamp: 0
      },
      {
        id: "5",
        index: 5,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg"
        },
        labels: {
          28: {
            id: "28",
            item: 5,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["28"],
            track: "1",
            order: 29,
            manual: false
          },
          74: {
            id: "74",
            item: 5,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["74"],
            track: "3",
            order: 75,
            manual: false
          },
          203: {
            id: "203",
            item: 5,
            sensors: [-1],
            type: "box2d",
            category: [0],
            attributes: {},
            parent: "",
            children: [],
            shapes: ["202"],
            track: "9",
            order: 204,
            manual: true
          }
        },
        shapes: {
          28: {
            id: "28",
            label: ["28"],
            shapeType: "rect",
            x1: 0,
            y1: 298.1679389312977,
            x2: 145.3482824427481,
            y2: 651.2977099236641
          },
          74: {
            id: "74",
            label: ["74"],
            shapeType: "rect",
            x1: 835.1192748091603,
            y1: 314.65648854961836,
            x2: 861.2261450381679,
            y2: 406.71755725190843
          },
          202: {
            id: "202",
            label: ["203"],
            shapeType: "rect",
            x1: 931.3024809160306,
            y1: 300.91603053435114,
            x2: 989.0124045801527,
            y2: 501.5267175572519
          }
        },
        timestamp: 0
      }
    ],
    tracks: {
      1: {
        id: "1",
        type: "box2d",
        labels: {
          0: "id23",
          1: "24",
          2: "25",
          3: "26",
          4: "27",
          5: "28"
        }
      },
      2: {
        id: "2",
        type: "box2d",
        labels: {
          0: "id46",
          1: "47",
          2: "48",
          3: "49"
        }
      },
      3: {
        id: "3",
        type: "box2d",
        labels: {
          0: "id69",
          1: "70",
          2: "71",
          3: "72",
          4: "73",
          5: "74"
        }
      },
      9: {
        id: "9",
        type: "box2d",
        labels: {
          5: "203"
        }
      }
    },
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
    id: "",
    select: {
      item: 0,
      labels: {},
      shapes: {},
      category: 0,
      attributes: {},
      labelType: 0,
      policyType: 0
    },
    layout: {
      toolbarWidth: 200,
      maxViewerConfigId: 0,
      maxPaneId: 0,
      rootPane: 0,
      panes: {
        0: {
          id: 0,
          viewerId: 0,
          parent: -1,
          hide: false,
          numHorizontalChildren: 0,
          numVerticalChildren: 0
        }
      }
    },
    viewerConfigs: []
  },
  session: {
    id: "",
    startTime: 0,
    itemStatuses: [
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      }
    ]
  }
}

export const emptyTrackingTask: DeepPartialState = {
  task: {
    config: {
      projectName: "Test3",
      itemType: "image",
      labelTypes: ["box2d"],
      label2DTemplates: {},
      taskSize: 8,
      handlerUrl: "label",
      pageTitle: "2D Bounding Box",
      instructionPage: "https://doc.scalabel.ai/instructions/bbox.html",
      bundleFile: "image_v2.js",
      categories: [
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
      ],
      attributes: [
        {
          name: "Occluded",
          toolType: AttributeToolType.SWITCH,
          tagText: "o",
          tagSuffixes: [],
          tagPrefix: "",
          values: [],
          buttonColors: []
        },
        {
          name: "Truncated",
          toolType: AttributeToolType.SWITCH,
          tagText: "t",
          tagSuffixes: [],
          tagPrefix: "",
          values: [],
          buttonColors: []
        },
        {
          name: "Traffic Color Light",
          toolType: AttributeToolType.LIST,
          tagText: "t",
          tagSuffixes: ["", "g", "y", "r"],
          tagPrefix: "",
          values: ["NA", "G", "Y", "R"],
          buttonColors: ["white", "green", "yellow", "red"]
        }
      ],
      taskId: "000000",
      tracking: true,
      policyTypes: ["linear_interpolation"],
      demoMode: false,
      autosave: true
    },
    status: {
      maxOrder: 223
    },
    items: [
      {
        id: "0",
        index: 0,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
      },
      {
        id: "1",
        index: 1,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
      },
      {
        id: "2",
        index: 2,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
      },
      {
        id: "3",
        index: 3,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
      },
      {
        id: "4",
        index: 4,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
      },
      {
        id: "5",
        index: 5,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
      },
      {
        id: "6",
        index: 6,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
      },
      {
        id: "7",
        index: 7,
        videoName: "",
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000106.jpg"
        },
        labels: {},
        shapes: {},
        timestamp: 0
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
    id: "",
    select: {
      item: 0,
      labels: {},
      shapes: {},
      category: 0,
      attributes: {},
      labelType: 0,
      policyType: 0
    },
    layout: {
      toolbarWidth: 200,
      maxViewerConfigId: 0,
      maxPaneId: 0,
      rootPane: 0,
      panes: {
        0: {
          id: 0,
          viewerId: 0,
          parent: -1,
          hide: false,
          numHorizontalChildren: 0,
          numVerticalChildren: 0
        }
      }
    },
    viewerConfigs: []
  },
  session: {
    id: "",
    startTime: 0,
    itemStatuses: [
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      },
      {
        sensorDataLoaded: {
          "-1": false
        }
      }
    ]
  }
}
