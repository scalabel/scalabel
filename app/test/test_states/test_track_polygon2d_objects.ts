import type { DeepPartialState } from "../../src/types/state"

export const testJson: DeepPartialState = {
  task: {
    config: {
      projectName: "testing",
      itemType: "image",
      labelTypes: ["polygon2d"],
      label2DTemplates: {},
      taskSize: 3,
      keyInterval: 1,
      handlerUrl: "label",
      pageTitle: "2D Segmentation Tracking",
      bundleFile: "image.js",
      categories: ["A", "B", "C"],
      treeCategories: [{ name: "A" }, { name: "B" }, { name: "C" }],
      attributes: [],
      taskId: "000000",
      tracking: true,
      policyTypes: ["linear_interpolation"],
      demoMode: false,
      autosave: true,
      bots: false
    },
    status: { maxOrder: 161 },
    items: [
      {
        id: "0",
        index: 0,
        videoName: "",
        urls: {
          "-1": "test_url_01"
        },
        labels: {},
        shapes: {},
        timestamp: 0,
        names: {}
      },
      {
        id: "1",
        index: 1,
        videoName: "",
        urls: {
          "-1": "test_url_02"
        },
        labels: {},
        shapes: {},
        timestamp: 0,
        names: {}
      },
      {
        id: "2",
        index: 2,
        videoName: "",
        urls: {
          "-1": "test_url_03"
        },
        labels: {},
        shapes: {},
        timestamp: 0,
        names: {}
      }
    ],
    tracks: {},
    sensors: { "-1": { id: -1, name: "default", type: "image" } },
    progress: { submissions: [] }
  },
  user: {
    id: "",
    select: {
      item: 0,
      labels: [],
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
        "0": {
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
      { sensorDataLoaded: { "-1": false } },
      { sensorDataLoaded: { "-1": false } },
      { sensorDataLoaded: { "-1": false } }
    ],
    trackLinking: false,
    status: 4,
    mode: 0,
    numUpdates: 0,
    alerts: [],
    info3D: { isBoxSpan: false, boxSpan: null, showGroundPlane: false }
  }
}
