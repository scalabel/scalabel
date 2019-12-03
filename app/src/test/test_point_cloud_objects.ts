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
      itemType: 'pointcloud',
      labelTypes: ['box3d'],
      policyTypes: ['linear_interpolation_box_3d'],
      tracking: false,
      taskSize: 5,
      handlerUrl: 'label3dv2',
      pageTitle: 'Scalabel Annotation',
      instructionPage: 'undefined',
      demoMode: false,
      bundleFile: 'label.js',
      categories: null,
      attributes: [
        {
          name: 'Occluded',
          toolType: 'switch',
          tagText: 'o'
        },
        {
          name: 'Truncated',
          toolType: 'switch',
          tagText: 't'
        }
      ],
      taskId: '000000',
      workerId: 'default_worker',
      startTime: 1539820189,
      submitTime: 0
    },
    items: [
      {
        id: 0,
        index: 0,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com' +
            '/scalabel-public/demo/luminar/1525401598139528987.ply'
        },
        labels: []
      },
      {
        id: 1,
        index: 1,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com' +
          '/scalabel-public/demo/luminar/1525401599138308593.ply'
        },
        labels: []
      },
      {
        id: 2,
        index: 2,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com' +
            '/scalabel-public/demo/luminar/1525401600135798773.ply'
        },
        labels: []
      },
      {
        id: 3,
        index: 3,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com' +
          '/scalabel-public/demo/luminar/1525401601134108834.ply'
        },
        labels: []
      },
      {
        id: 4,
        index: 4,
        urls: {
          '-1': 'https://s3-us-west-2.amazonaws.com' +
          '/scalabel-public/demo/luminar/1525401602133158775.ply'
        },
        labels: []
      }
    ],
    tracks: {},
    sensors: {
      '-1': {
        id: -1,
        name: 'default',
        type: 'pointcloud'
      }
    }
  }
}
