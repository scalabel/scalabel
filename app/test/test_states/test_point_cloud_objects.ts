import { AttributeToolType } from "../../src/const/common"
import { DeepPartialState } from "../../src/types/state"

export const testJson: DeepPartialState = {
  session: {
    itemStatuses: [
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} },
      { sensorDataLoaded: {} }
    ],
    startTime: 1539820189
  },
  task: {
    config: {
      projectName: "Redux0",
      itemType: "pointcloud",
      labelTypes: ["box3d"],
      label2DTemplates: {},
      policyTypes: ["linear_interpolation_box_3d"],
      tracking: false,
      taskSize: 5,
      handlerUrl: "label3dv2",
      pageTitle: "Scalabel Annotation",
      instructionPage: "undefined",
      demoMode: false,
      attributes: [
        {
          name: "Occluded",
          toolType: AttributeToolType.SWITCH,
          tagText: "o"
        },
        {
          name: "Truncated",
          toolType: AttributeToolType.SWITCH,
          tagText: "t"
        }
      ],
      taskId: "000000"
    },
    items: [
      {
        id: "id0",
        index: 0,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com" +
            "/scalabel-public/demo/luminar/1525401598139528987.ply"
        },
        labels: {}
      },
      {
        id: "1",
        index: 1,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com" +
            "/scalabel-public/demo/luminar/1525401599138308593.ply"
        },
        labels: {}
      },
      {
        id: "id2",
        index: 2,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com" +
            "/scalabel-public/demo/luminar/1525401600135798773.ply"
        },
        labels: {}
      },
      {
        id: "id3",
        index: 3,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com" +
            "/scalabel-public/demo/luminar/1525401601134108834.ply"
        },
        labels: {}
      },
      {
        id: "abc",
        index: 4,
        urls: {
          "-1":
            "https://s3-us-west-2.amazonaws.com" +
            "/scalabel-public/demo/luminar/1525401602133158775.ply"
        },
        labels: {}
      }
    ],
    tracks: {},
    sensors: {
      "-1": {
        id: -1,
        name: "default",
        type: "pointcloud"
      }
    },
    progress: {
      submissions: []
    }
  }
}
