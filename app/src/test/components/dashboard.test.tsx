import { MuiThemeProvider } from '@material-ui/core/styles'
import { cleanup, render } from '@testing-library/react'
import React from 'react'
import Dashboard, {
  DashboardContents, StyledHeader, StyledSidebar
} from '../../js/components/dashboard'
import { myTheme } from '../../js/styles/theme'

afterEach(cleanup)
describe('Test dashboard functionality', () => {
  test('Correct table values', () => {
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <Dashboard dashboardContents={sampleDashboardContents} />
      </MuiThemeProvider>)
    sampleDashboardContents.taskMetaDatas.forEach((value, index) => {
      const numLabeledImages = getByTestId('num-labeled-images-'
        + index.toString()).innerHTML
      const numLabels = getByTestId('num-labels-' + index.toString()).innerHTML
      expect(numLabeledImages).toBe(value.numLabeledItems)
      expect(numLabels).toBe(value.numLabels)
    })
  })
  test('Correct totals', () => {
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <Dashboard dashboardContents={sampleDashboardContents} />
      </MuiThemeProvider>)
    const totalTasks = getByTestId('total-tasks').firstElementChild!.innerHTML
    const totalLabels = getByTestId('total-labels').firstElementChild!.innerHTML
    expect(totalTasks).toBe('3')
    expect(totalLabels).toBe('6')
  })
  test('Correct task urls', () => {
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <Dashboard dashboardContents={sampleDashboardContents} />
      </MuiThemeProvider>)
    sampleDashboardContents.taskMetaDatas.forEach((value, index) => {
      const taskLinkElem = getByTestId('task-link-' +
        index.toString()) as HTMLAnchorElement
      const url = taskLinkElem.href
      expect(url).toContain(value.handlerUrl +
        '?project_name=' +
        sampleDashboardContents.projectMetaData.name +
        '&task_index=' + index)
    })
  })
  describe('Test dashboard sidebar links', () => {
    // test('Correct non-tag url', () => {
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
    test('Correct tag url', () => {
      const tagProjectMetadata = { ...sampleDashboardContents.projectMetaData }
      tagProjectMetadata.labelTypes = ['tag']
      const { getByTestId } = render(
        <MuiThemeProvider theme={myTheme}>
          <StyledSidebar projectMetaData=
            {tagProjectMetadata}
          />
        </MuiThemeProvider>)
      const exportElem = getByTestId('export-link') as HTMLLinkElement
      const exportURL = exportElem.href
      expect(exportURL).toContain('V2')
    })
    describe('Test not rendering elements if vendor', () => {
      test('Hiding header elements', () => {
        const { queryByTestId } = render(
          <MuiThemeProvider theme={myTheme}>
            <StyledHeader
              totalTaskLabeled={0}
              totalLabels={0}
              vendor={true}
            />
          </MuiThemeProvider>)
        expect(queryByTestId('total-tasks')).toBeNull()
        expect(queryByTestId('total-labels')).toBeNull()
      })
      test('Hiding sidebar elements', () => {
        const projectMetaData = sampleDashboardContents.projectMetaData
        const { queryByTestId } = render(
          <MuiThemeProvider theme={myTheme}>
            <StyledSidebar
              projectMetaData={projectMetaData}
              vendor={true}
            />
          </MuiThemeProvider>)
        expect(queryByTestId('export-link')).toBeNull()
        expect(queryByTestId('download-link')).toBeNull()
      })
    })
  })
})

const sampleDashboardContents: DashboardContents = {
  projectMetaData: {
    name: 'test',
    itemType: 'image',
    labelTypes: ['box2d'],
    taskSize: 4,
    numLeafCategories: 0,
    numItems: 5,
    numAttributes: 2
  },
  taskMetaDatas: [{
    numLabeledItems: '5',
    numLabels: '3',
    submitted: true,
    handlerUrl: 'url'
  }, {
    numLabeledItems: '1',
    numLabels: '2',
    submitted: true,
    handlerUrl: 'url'
  }, {
    numLabeledItems: '1',
    numLabels: '1',
    submitted: true,
    handlerUrl: 'url'
  }, {
    numLabeledItems: '0',
    numLabels: '0',
    submitted: true,
    handlerUrl: 'url'
  }]
}
