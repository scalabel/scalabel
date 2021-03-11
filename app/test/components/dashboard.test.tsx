import { ThemeProvider } from "@material-ui/core/styles"
import { cleanup, render } from "@testing-library/react"
import React from "react"

import Dashboard, {
  DashboardContents,
  StyledHeader,
  StyledSidebar
} from "../../src/components/dashboard"
import { scalabelTheme } from "../../src/styles/theme"
import { SubmitData } from "../../src/types/state"

afterEach(cleanup)
describe("Test dashboard functionality", () => {
  test("Correct table values", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Dashboard dashboardContents={sampleDashboardContents} />
      </ThemeProvider>
    )
    sampleDashboardContents.taskMetaDatas.forEach((value, index) => {
      const numLabeledImages = getByTestId(
        "num-labeled-images-" + index.toString()
      ).innerHTML
      const numLabels = getByTestId("num-labels-" + index.toString()).innerHTML
      expect(numLabeledImages).toBe(value.numLabeledItems)
      expect(numLabels).toBe(value.numLabels)
    })
  })
  test("Correct totals", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Dashboard dashboardContents={sampleDashboardContents} />
      </ThemeProvider>
    )
    const totalTaskElement = getByTestId("total-tasks").firstElementChild
    const totalLabelElement = getByTestId("total-labels").firstElementChild
    expect(totalTaskElement).not.toBeNull()
    expect(totalLabelElement).not.toBeNull()
    if (totalTaskElement !== null && totalLabelElement !== null) {
      const totalTasks = totalTaskElement.innerHTML
      const totalLabels = totalLabelElement.innerHTML
      expect(totalTasks).toBe("3")
      expect(totalLabels).toBe("6")
    }
  })
  test("Correct task urls", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Dashboard dashboardContents={sampleDashboardContents} />
      </ThemeProvider>
    )
    sampleDashboardContents.taskMetaDatas.forEach((value, index) => {
      const taskLinkElem = getByTestId(
        "task-link-" + index.toString()
      ) as HTMLAnchorElement
      const url = taskLinkElem.href
      expect(url).toContain(
        `${value.handlerUrl}?project_name=${sampleDashboardContents.projectMetaData.name}&task_index=${index}`
      )
    })
  })
  describe("Test dashboard sidebar links", () => {
    // Test('Correct non-tag url', () => {
    //   const { getByTestId } = render(
    //     <MuiThemeProvider theme={myTheme}>
    //       <StyledSidebar projectMetaData=
    //         {sampleDashboardContents.projectMetaData}
    //       />
    //     </MuiThemeProvider>)
    //   const exportElem = getByTestId('export-link') as HTMLLinkElement
    //   const exportURL = exportElem.href
    //   expect(exportURL).not.toContain('V2')
    // })
    describe("Test not rendering elements if vendor", () => {
      test("Hiding header elements", () => {
        const { queryByTestId } = render(
          <ThemeProvider theme={scalabelTheme}>
            <StyledHeader
              totalTaskLabeled={0}
              totalLabels={0}
              numUsers={0}
              vendor={true}
            />
          </ThemeProvider>
        )
        expect(queryByTestId("total-tasks")).toBeNull()
        expect(queryByTestId("total-labels")).toBeNull()
      })
      test("Hiding sidebar elements", () => {
        const projectMetaData = sampleDashboardContents.projectMetaData
        const { queryByTestId } = render(
          <ThemeProvider theme={scalabelTheme}>
            <StyledSidebar projectMetaData={projectMetaData} vendor={true} />
          </ThemeProvider>
        )
        expect(queryByTestId("export-link")).toBeNull()
        expect(queryByTestId("download-link")).toBeNull()
      })
    })
  })
})
const submitData: SubmitData[] = [
  {
    time: 0,
    user: "user"
  }
]

const sampleDashboardContents: DashboardContents = {
  projectMetaData: {
    name: "test",
    itemType: "image",
    labelTypes: ["box2d"],
    taskSize: 4,
    numLeafCategories: 0,
    numItems: 5,
    numAttributes: 2
  },
  taskMetaDatas: [
    {
      numLabeledItems: "5",
      numLabels: "3",
      submissions: submitData,
      handlerUrl: "url"
    },
    {
      numLabeledItems: "1",
      numLabels: "2",
      submissions: submitData,
      handlerUrl: "url"
    },
    {
      numLabeledItems: "1",
      numLabels: "1",
      submissions: submitData,
      handlerUrl: "url"
    },
    {
      numLabeledItems: "0",
      numLabels: "0",
      submissions: submitData,
      handlerUrl: "url"
    }
  ],
  numUsers: 0
}
