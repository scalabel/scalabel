export const testJson = {
  task: {
    config: {
      assignmentId: 'e6015077-aad9-4e60-a5ed-dbccf931a049',
      projectName: 'Redux0',
      itemType: 'image',
      labelTypes: ['box2d', 'tag'],
      taskSize: 5,
      handlerUrl: 'label2dv2',
      pageTitle: 'Image Tagging Labeling Tool',
      instructionPage: 'undefined',
      demoMode: false,
      bundleFile: 'image_v2.js',
      categories: null,
      attributes: [
        {
          name: 'Weather',
          toolType: 'list',
          tagText: '',
          tagPrefix: 'w',
          tagSuffixes: [
            '',
            'r',
            's',
            'c',
            'o',
            'p',
            'f'
          ],
          values: [
            'NA',
            'Rainy',
            'Snowy',
            'Clear',
            'Overcast',
            'Partly Cloudy',
            'Foggy'
          ],
          buttonColors: [
            'white',
            'white',
            'white',
            'white',
            'white',
            'white',
            'white'
          ]
        },
        {
          name: 'Scene',
          toolType: 'list',
          tagText: '',
          tagPrefix: 's',
          tagSuffixes: [
            '',
            't',
            'r',
            'p',
            'c',
            'g',
            'h'
          ],
          values: [
            'NA',
            'Tunnel',
            'Residential',
            'Parking Lot',
            'City Street',
            'Gas Stations',
            'Highway'
          ],
          buttonColors: [
            'white',
            'white',
            'white',
            'white',
            'white',
            'white',
            'white'
          ]
        },
        {
          name: 'Timeofday',
          toolType: 'list',
          tagText: '',
          tagPrefix: 't',
          tagSuffixes: ['', 'day', 'n', 'daw'],
          values: [
            'NA',
            'Daytime',
            'Night',
            'Dawn/Dusk'
          ],
          buttonColors: [
            'white',
            'white',
            'white',
            'white'
          ]
        }
      ],
      taskId: '000000',
      workerId: 'default_worker',
      startTime: 1539820189,
      submitTime: 0
    },
    current: {
      item: -1,
      label: -1,
      labelType: 0,
      maxLabelId: -1,
      maxShapeId: -1,
      maxOrder: -1
    },
    items: [
      {
        id: 0,
        index: 0,
        url: 'https://s3-us-west-2.amazonaws.com/' +
          'scalabel-public/demo/frames/intersection-0000051.jpg',
        active: false,
        loaded: false,
        labels: []
      },
      {
        id: 1,
        index: 1,
        url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
          'demo/frames/intersection-0000052.jpg',
        active: false,
        loaded: false,
        labels: []
      },
      {
        id: 2,
        index: 2,
        url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
          'demo/frames/intersection-0000053.jpg',
        active: false,
        loaded: false,
        labels: []
      },
      {
        id: 3,
        index: 3,
        url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
          'demo/frames/intersection-0000054.jpg',
        active: false,
        loaded: false,
        labels: []
      },
      {
        id: 4,
        index: 4,
        url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
          'demo/frames/intersection-0000055.jpg',
        active: false,
        loaded: false,
        labels: []
      }
    ],
    labels: {},
    tracks: {},
    shapes: {}
  },
  actions: []
}

const dummyViewerConfig = {
  imageWidth: 800,
  imageHeight: 800,
  viewScale: 1,
  viewOffsetX: 0,
  viewOffsetY: 0
}

/**
 * make a dummy item for testing
 */
function makeItem (
  id: number,
  labels: object,
  shapes: object
): object {
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
  type: 'bbox',
  category: [],
  attributes: {},
  parent: 1,
  children: [],
  shapes: [
    2
  ],
  selectedShape: 2,
  state: 2,
  order: 2
}

const dummyShapes = {
  0: {
    id: 0,
    label: [0],
    shape: {
      x1: 0, y1: 0, x2: 1, y2: 1
    }
  },
  1: {
    id: 1,
    label: [1],
    shape: { x1: 2, y1: 2, x2: 3, y2: 3 }
  }
}

export const autoSaveTestJson = {
  items: [
    makeItem(0, dummyLabels, dummyShapes),
    makeItem(1, {}, {})
  ],
  current: {
    item: 0,
    maxLabelId: 1,
    maxShapeId: 1,
    maxOrder: 1
  }
}
