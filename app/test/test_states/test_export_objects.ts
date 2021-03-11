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
      box3d: null,
      plane3d: null,
      customs: {}
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
      box3d: null,
      plane3d: null,
      customs: {}
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
      box3d: null,
      plane3d: null,
      customs: {}
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
        box3d: null,
        plane3d: null,
        customs: {}
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
        box3d: null,
        plane3d: null,
        customs: {}
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
        box3d: null,
        plane3d: null,
        customs: {}
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
      box3d: null,
      plane3d: null,
      customs: {}
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
      box3d: null,
      plane3d: null,
      customs: {}
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
      box3d: null,
      plane3d: null,
      customs: {}
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
        box3d: null,
        plane3d: null,
        customs: {}
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
        box3d: null,
        plane3d: null,
        customs: {}
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
        box3d: null,
        plane3d: null,
        customs: {}
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
