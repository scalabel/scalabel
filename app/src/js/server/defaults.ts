import { AttributeToolType } from '../const/common'
import { StorageType } from '../const/config'
import { ServerConfig } from '../types/config'

/* default config */
export const serverConfig: ServerConfig = {
  http: {
    port:  8686
  },
  storage: {
    type: StorageType.LOCAL,
    data: './',
    itemDir: ''
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
    timeout: 3600,
    writebackTime: 600,
    writebackActions: 32,
    port: 6379
  },
  bot: {
    on: false,
    host: 'http://0.0.0.0',
    port: 8080
  }
}

/* default categories when file is missing and label is box2D or box3D */
export const boxCategories = [
  'person',
  'rider',
  'car',
  'truck',
  'bus',
  'train',
  'motor',
  'bike',
  'traffic sign',
  'traffic light'
]

/* default categories when file is missing and label is polyline2d */
export const polyline2DCategories = [
  'road curb',
  'double white',
  'double yellow',
  'double other',
  'single white',
  'single yellow',
  'single other',
  'crosswalk'
]

// TODO: add default seg2d categories once nested categories are supported

/* default attributes when file is missing and label is box2D */
export const box2DAttributes = [
  {
    name: 'Occluded',
    toolType: AttributeToolType.SWITCH,
    tagText: 'o',
    tagSuffixes: [],
    tagPrefix: '',
    values: [],
    buttonColors: []
  },
  {
    name: 'Truncated',
    toolType: AttributeToolType.SWITCH,
    tagText: 't',
    tagSuffixes: [],
    tagPrefix: '',
    values: [],
    buttonColors: []
  },
  {
    name: 'Traffic Color Light',
    toolType: AttributeToolType.LIST,
    tagText: 't',
    tagSuffixes: ['', 'g', 'y', 'r'],
    tagPrefix: '',
    values: ['NA', 'G', 'Y', 'R'],
    buttonColors: ['white', 'green', 'yellow', 'red']
  }
]

/* default attributes when file is missing and no other defaults exist */
export const dummyAttributes = [{
  name: '',
  toolType: AttributeToolType.NONE,
  tagText: '',
  tagSuffixes: [],
  values: [],
  tagPrefix: '',
  buttonColors: []
}]
