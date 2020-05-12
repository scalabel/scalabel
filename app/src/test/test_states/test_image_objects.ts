export const testJson = {
  session: {
    itemStatuses: [
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} }
    ]
  },
  task: {
    config: {
      assignmentId: 'e6015077-aad9-4e60-a5ed-dbccf931a049',
      projectName: 'Redux0',
      itemType: 'image',
      labelTypes:
        ['box2d', 'polygon2d', 'tag', 'polyline2d', 'basicHumanPose'],
      label2DTemplates: {
        basicHumanPose: {
          name: 'Basic Human Pose',
          nodes: [
              { name: 'Head', x: 5, y: 5, color: [0, 0, 255] },
              { name: 'Neck', x: 5, y: 10, color: [0, 0, 255] },
              { name: 'Left Shoulder', x: 1, y: 10, color: [0, 255, 0] },
              { name: 'Right Shoulder', x: 9, y: 10, color: [255, 0, 0] },
              { name: 'Left Elbow', x: 1, y: 13, color: [0, 255, 0] },
              { name: 'Right Elbow', x: 9, y: 13, color: [255, 0, 0] },
              { name: 'Left Wrist', x: 1, y: 17, color: [0, 255, 0] },
              { name: 'Right Wrist', x: 9, y: 17, color: [255, 0, 0] },
              { name: 'Pelvis', x: 5, y: 22, color: [0, 0, 255] },
              { name: 'Left Knee', x: 3, y: 30, color: [0, 255, 0] },
              { name: 'Right Knee', x: 7, y: 30, color: [255, 0, 0] },
              { name: 'Left Foot', x: 3, y: 40, color: [0, 255, 0] },
              { name: 'Right Foot', x: 7, y: 40, color: [255, 0, 0] }
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
      policyTypes: ['linear_interpolation_box_2d', 'linear_interpolation_polygon'],
      tracking: false,
      taskSize: 5,
      handlerUrl: 'label2dv2',
      pageTitle: 'Image Tagging Labeling Tool',
      instructionPage: 'undefined',
      demoMode: false,
      autosave: false,
      bundleFile: 'image_v2.js',
      categories: ['1', '2', '3'],
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
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com/' +
            'scalabel-public/demo/frames/intersection-0000051.jpg'
        },
        labels: []
      },
      {
        id: 1,
        index: 1,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
            'demo/frames/intersection-0000052.jpg'
        },
        labels: []
      },
      {
        id: 2,
        index: 2,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
            'demo/frames/intersection-0000053.jpg'
        },
        labels: []
      },
      {
        id: 3,
        index: 3,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
            'demo/frames/intersection-0000054.jpg'
        },
        labels: []
      },
      {
        id: 4,
        index: 4,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com/scalabel-public/' +
            'demo/frames/intersection-0000055.jpg'
        },
        labels: []
      }
    ],
    tracks: {},
    sensors: {
      '-1': {
        id: -1,
        name: 'default',
        type: 'image'
      }
    }
  }
}
