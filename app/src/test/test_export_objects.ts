import { ItemExport } from '../js/functional/bdd_types'

export const sampleItemExport: ItemExport = {
  name: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg',
  url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg',
  videoName: '',
  attributes: {},
  timestamp: 1570506897,
  index: 0,
  labels: [
    {
      id: 1,
      category: 'person',
      attributes: {
        'Occluded': false,
        'Truncated': false,
        'Traffic Light Color': [
          'NA'
        ]
      },
      manualShape: true,
      box2d: {
        x1: 280.00968616094616,
        x2: 450.53600195041986,
        y1: 152.57617728531858,
        y2: 446.75900277008316
      },
      poly2d: null,
      box3d: null
    },
    {
      id: 2,
      category: 'rider',
      attributes: {
        'Occluded': true,
        'Truncated': false,
        'Traffic Light Color': [
          'R'
        ]
      },
      manualShape: true,
      box2d: {
        x1: 405.66065569003206,
        x2: 618.0706279892011,
        y1: 73.79501385041551,
        y2: 215.4016620498615
      },
      poly2d: null,
      box3d: null
    },
    {
      id: 3,
      category: 'truck',
      attributes: {
        'Occluded': false,
        'Truncated': true,
        'Traffic Light Color': [
          'G', 'Y'
        ]
      },
      manualShape: true,
      box2d: {
        x1: 648.9847554130238,
        x2: 915.245143224658,
        y1: 106.70360110803324,
        y2: 329.0858725761773
      },
      poly2d: null,
      box3d: null
    }
  ]
}

export const sampleStateExport: ItemExport[] =
  [
    {
      name: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg',
      url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg',
      videoName: '',
      attributes: {},
      timestamp: 1570506897,
      index: 0,
      labels: [
        {
          id: 1,
          category: 'person',
          attributes: {
            'Occluded': false,
            'Truncated': false,
            'Traffic Light Color': [
              'NA'
            ]
          },
          manualShape: true,
          box2d: {
            x1: 280.00968616094616,
            x2: 450.53600195041986,
            y1: 152.57617728531858,
            y2: 446.75900277008316
          },
          poly2d: null,
          box3d: null
        },
        {
          id: 2,
          category: 'rider',
          attributes: {
            'Occluded': true,
            'Truncated': false,
            'Traffic Light Color': [
              'R'
            ]
          },
          manualShape: true,
          box2d: {
            x1: 405.66065569003206,
            x2: 618.0706279892011,
            y1: 73.79501385041551,
            y2: 215.4016620498615
          },
          poly2d: null,
          box3d: null
        },
        {
          id: 3,
          category: 'truck',
          attributes: {
            'Occluded': false,
            'Truncated': true,
            'Traffic Light Color': [
              'G', 'Y'
            ]
          },
          manualShape: true,
          box2d: {
            x1: 648.9847554130238,
            x2: 915.245143224658,
            y1: 106.70360110803324,
            y2: 329.0858725761773
          },
          poly2d: null,
          box3d: null
        }
      ]
    },
    {
      name: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg',
      url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000102.jpg',
      videoName: '',
      attributes: {},
      timestamp: 1570506897,
      index: 1,
      labels: []
    },
    {
      name: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg',
      url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000103.jpg',
      videoName: '',
      attributes: {},
      timestamp: 1570506897,
      index: 2,
      labels: []
    },
    {
      name: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg',
      url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000104.jpg',
      videoName: '',
      attributes: {},
      timestamp: 1570506897,
      index: 3,
      labels: []
    },
    {
      name: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg',
      url: 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000105.jpg',
      videoName: '',
      attributes: {},
      timestamp: 1570506897,
      index: 4,
      labels: []
    }
  ]
