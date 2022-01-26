import { ItemExport } from "../../src/types/export"

export const sampleItemExportImage: ItemExport = {
  name:
    "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
  url:
    "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
  videoName: "b",
  attributes: {},
  timestamp: 0,
  sensor: -1,
  labels: [
    {
      id: "1",
      category: "person",
      attributes: {},
      manualShape: true,
      box2d: {
        x1: 459.5110712024974,
        y1: 276.2326869806094,
        x2: 752.6966667703645,
        y2: 400.88642659279776
      },
      poly2d: null,
      box3d: null
    },
    {
      id: "2",
      category: "person",
      attributes: {},
      manualShape: true,
      box2d: {
        x1: 746.7132872689795,
        y1: 137.61772853185596,
        x2: 1018.9570545819988,
        y2: 294.1828254847645
      },
      poly2d: null,
      box3d: null
    },
    {
      id: "longLabelId",
      category: "person",
      attributes: {},
      manualShape: true,
      box2d: {
        x1: 317.9044230030514,
        y1: 192.46537396121883,
        x2: 676.9071930861539,
        y2: 411.85595567867034
      },
      poly2d: null,
      box3d: null
    }
  ]
}

export const sampleStateExportImage: ItemExport[] = [
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 0,
    sensor: -1,
    labels: [
      {
        id: "1",
        category: "person",
        attributes: {},
        manualShape: true,
        box2d: {
          x1: 459.5110712024974,
          y1: 276.2326869806094,
          x2: 752.6966667703645,
          y2: 400.88642659279776
        },
        poly2d: null,
        box3d: null
      },
      {
        id: "2",
        category: "person",
        attributes: {},
        manualShape: true,
        box2d: {
          x1: 746.7132872689795,
          y1: 137.61772853185596,
          x2: 1018.9570545819988,
          y2: 294.1828254847645
        },
        poly2d: null,
        box3d: null
      },
      {
        id: "longLabelId",
        category: "person",
        attributes: {},
        manualShape: true,
        box2d: {
          x1: 317.9044230030514,
          y1: 192.46537396121883,
          x2: 676.9071930861539,
          y2: 411.85595567867034
        },
        poly2d: null,
        box3d: null
      }
    ]
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000052.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000052.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 1,
    labels: [],
    sensor: -1
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000053.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000053.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 2,
    labels: [],
    sensor: -1
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000054.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000054.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 3,
    labels: [],
    sensor: -1
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000055.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000055.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 4,
    labels: [],
    sensor: -1
  }
]

export const sampleItemExportImagePolygon: ItemExport = {
  name:
    "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
  url:
    "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
  videoName: "b",
  attributes: {},
  timestamp: 0,
  sensor: -1,
  labels: [
    {
      id: "0",
      category: "person",
      attributes: {},
      manualShape: true,
      box2d: null,
      poly2d: [
        {
          vertices: [
            [100, 100],
            [200, 100],
            [200, 200]
          ],
          types: "LLL",
          closed: true
        }
      ],
      box3d: null
    },
    {
      id: "1",
      category: "person",
      attributes: {},
      manualShape: true,
      box2d: null,
      poly2d: [
        {
          types: "LLCCLCC",
          vertices: [
            [300, 300],
            [400, 400],
            [350, 300],
            [325, 200],
            [400, 100],
            [375, 125],
            [325, 250]
          ],
          closed: true
        }
      ],
      box3d: null
    },
    {
      id: "longPolyLabelId",
      category: "person",
      attributes: {},
      manualShape: true,
      box2d: null,
      poly2d: [
        {
          types: "LLLCC",
          vertices: [
            [400, 400],
            [800, 100],
            [100, 100],
            [125, 120],
            [325, 220]
          ],
          closed: true
        }
      ],
      box3d: null
    }
  ]
}

export const sampleStateExportImagePolygon: ItemExport[] = [
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 0,
    sensor: -1,
    labels: [
      {
        id: "0",
        category: "person",
        attributes: {},
        manualShape: true,
        box2d: null,
        poly2d: [
          {
            types: "LLL",
            vertices: [
              [100, 100],
              [200, 100],
              [200, 200]
            ],
            closed: true
          }
        ],
        box3d: null
      },
      {
        id: "1",
        category: "person",
        attributes: {},
        manualShape: true,
        box2d: null,
        poly2d: [
          {
            types: "LLCCLCC",
            vertices: [
              [300, 300],
              [400, 400],
              [350, 300],
              [325, 200],
              [400, 100],
              [375, 125],
              [325, 250]
            ],
            closed: true
          }
        ],
        box3d: null
      },
      {
        id: "longPolyLabelId",
        category: "person",
        attributes: {},
        manualShape: true,
        box2d: null,
        poly2d: [
          {
            types: "LLLCC",
            vertices: [
              [400, 400],
              [800, 100],
              [100, 100],
              [125, 120],
              [325, 220]
            ],
            closed: true
          }
        ],
        box3d: null
      }
    ]
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000052.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000052.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 1,
    labels: [],
    sensor: -1
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000053.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000053.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 2,
    labels: [],
    sensor: -1
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000054.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000054.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 3,
    labels: [],
    sensor: -1
  },
  {
    name:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000055.jpg",
    url:
      "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000055.jpg",
    videoName: "b",
    attributes: {},
    timestamp: 4,
    labels: [],
    sensor: -1
  }
]

export const sampleItemExportImageTagging: ItemExport = {
  name:
    "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg",
  url:
    "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg",
  videoName: "",
  attributes: {
    Weather: "Rainy",
    Scene: "City Street",
    Timeofday: "Night"
  },
  timestamp: 0,
  sensor: -1,
  labels: []
}

export const sampleItemExportImage3dBox: ItemExport = {
  name:
    "http://localhost:8686/items/kitti/tracking/training/image_02/0001/000000.png",
  url:
    "http://localhost:8686/items/kitti/tracking/training/image_02/0001/000000.png",
  videoName: "",
  timestamp: 0,
  attributes: {},
  labels: [
    {
      id: "0",
      category: "car",
      attributes: {},
      manualShape: false,
      box2d: null,
      poly2d: null,
      box3d: {
        location: [2.921483, 0.755883, 6.348542],
        dimension: [1.50992, 1.85, 4.930564],
        orientation: [0, -1.570796, 0]
      }
    },
    {
      id: "1",
      category: "car",
      attributes: {},
      manualShape: false,
      box2d: null,
      poly2d: null,
      box3d: {
        location: [2.994469, 0.8304805, 13.169745],
        dimension: [1.404795, 1.612032, 3.772344],
        orientation: [0, -1.570796, 0]
      }
    },
    {
      id: "2",
      category: "car",
      attributes: {},
      manualShape: false,
      box2d: null,
      poly2d: null,
      box3d: {
        location: [2.908125, 0.8767944999999999, 19.299001],
        dimension: [1.413269, 1.567278, 3.158158],
        orientation: [0, -1.511817, 0]
      }
    },
    {
      id: "3",
      category: "car",
      attributes: {},
      manualShape: false,
      box2d: null,
      poly2d: null,
      box3d: {
        location: [-6.037999, 1.4392370000000003, 23.712843],
        dimension: [1.527328, 1.555003, 3.57675],
        orientation: [0, 1.587202, 0]
      }
    },
    {
      id: "4",
      category: "car",
      attributes: {},
      manualShape: false,
      box2d: null,
      poly2d: null,
      box3d: {
        location: [-6.310451, 1.7865565, 46.486705],
        dimension: [1.417371, 1.540476, 3.504344],
        orientation: [0, 1.5615160000000001, 0]
      }
    },
    {
      id: "5",
      category: "car",
      attributes: {},
      manualShape: false,
      box2d: null,
      poly2d: null,
      box3d: {
        location: [2.822686, 1.2086425, 50.003616],
        dimension: [1.512623, 1.746309, 3.775171],
        orientation: [0, -1.572189, 0]
      }
    },
    {
      id: "6",
      category: "car",
      attributes: {},
      manualShape: false,
      box2d: null,
      poly2d: null,
      box3d: {
        location: [-6.285505, 1.7774785, 52.066669],
        dimension: [1.360295, 1.513462, 4.017021],
        orientation: [0, 1.583558, 0]
      }
    }
  ],
  sensor: -1,
  intrinsics: {
    focal: [721.5377197265625, 721.5377197265625],
    center: [609.559326171875, 172.85400390625]
  },
  extrinsics: {
    location: [-0.5327253937721252, 0.0, 0.0],
    rotation: [-1.5723664220969806, 0.07224191126404556, -1.8406363281435645]
  }
}

export const sampleStateExportImage3dBox: ItemExport[] = [
  {
    name:
      "http://localhost:8686/items/kitti/tracking/training/image_02/0001/000000.png",
    url:
      "http://localhost:8686/items/kitti/tracking/training/image_02/0001/000000.png",
    videoName: "",
    timestamp: 0,
    attributes: {},
    labels: [
      {
        id: "0",
        category: "car",
        attributes: {},
        manualShape: false,
        box2d: null,
        poly2d: null,
        box3d: {
          location: [2.921483, 0.755883, 6.348542],
          dimension: [1.50992, 1.85, 4.930564],
          orientation: [0, -1.570796, 0]
        }
      },
      {
        id: "1",
        category: "car",
        attributes: {},
        manualShape: false,
        box2d: null,
        poly2d: null,
        box3d: {
          location: [2.994469, 0.8304805, 13.169745],
          dimension: [1.404795, 1.612032, 3.772344],
          orientation: [0, -1.570796, 0]
        }
      },
      {
        id: "2",
        category: "car",
        attributes: {},
        manualShape: false,
        box2d: null,
        poly2d: null,
        box3d: {
          location: [2.908125, 0.8767944999999999, 19.299001],
          dimension: [1.413269, 1.567278, 3.158158],
          orientation: [0, -1.511817, 0]
        }
      },
      {
        id: "3",
        category: "car",
        attributes: {},
        manualShape: false,
        box2d: null,
        poly2d: null,
        box3d: {
          location: [-6.037999, 1.4392370000000003, 23.712843],
          dimension: [1.527328, 1.555003, 3.57675],
          orientation: [0, 1.587202, 0]
        }
      },
      {
        id: "4",
        category: "car",
        attributes: {},
        manualShape: false,
        box2d: null,
        poly2d: null,
        box3d: {
          location: [-6.310451, 1.7865565, 46.486705],
          dimension: [1.417371, 1.540476, 3.504344],
          orientation: [0, 1.5615160000000001, 0]
        }
      },
      {
        id: "5",
        category: "car",
        attributes: {},
        manualShape: false,
        box2d: null,
        poly2d: null,
        box3d: {
          location: [2.822686, 1.2086425, 50.003616],
          dimension: [1.512623, 1.746309, 3.775171],
          orientation: [0, -1.572189, 0]
        }
      },
      {
        id: "6",
        category: "car",
        attributes: {},
        manualShape: false,
        box2d: null,
        poly2d: null,
        box3d: {
          location: [-6.285505, 1.7774785, 52.066669],
          dimension: [1.360295, 1.513462, 4.017021],
          orientation: [0, 1.583558, 0]
        }
      }
    ],
    sensor: -1,
    intrinsics: {
      focal: [721.5377197265625, 721.5377197265625],
      center: [609.559326171875, 172.85400390625]
    },
    extrinsics: {
      location: [-0.5327253937721252, 0.0, 0.0],
      rotation: [-1.5723664220969806, 0.07224191126404556, -1.8406363281435645]
    }
  }
]
