import { ThemeProvider, withStyles } from "@material-ui/styles"
import { cleanup, fireEvent, render } from "@testing-library/react"
import React from "react"

import CreateForm from "../../src/components/create_form"
import { formStyle } from "../../src/styles/create"
import { scalabelTheme } from "../../src/styles/theme"

/* eslint-disable */
afterAll(() => {
  ;(window as any).XMLHttpRequest = oldXMLHttpRequest
  window.alert = oldAlert
})

const oldXMLHttpRequest = (window as any).XMLHttpRequest
const oldAlert = window.alert

const xhrMockClass = {
  open: jest.fn(),
  send: jest.fn(),
  onreadystatechange: jest.fn(),
  readyState: 4,
  response: ""
}
;(window as any).XMLHttpRequest = jest
  .fn()
  .mockImplementation(() => xhrMockClass)
window.alert = jest.fn()
/* eslint-enable */

afterEach(cleanup)
describe("Test different label options", () => {
  test("Proper page titles", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const pageTitle = getByTestId("page-title") as HTMLInputElement
    const select = getByTestId("label-type") as HTMLSelectElement
    expect(pageTitle.value).toBeFalsy()
    fireEvent.change(select, { target: { value: "tag" } })
    expect(pageTitle.value).toBe("Image Tagging")
    fireEvent.change(select, { target: { value: "box2d" } })
    expect(pageTitle.value).toBe("2D Bounding Box")
    fireEvent.change(select, { target: { value: "polygon2d" } })
    expect(pageTitle.value).toBe("2D Segmentation")
    fireEvent.change(select, { target: { value: "polyline2d" } })
    expect(pageTitle.value).toBe("2D Lane")
    // Box 3D is disabled for now
    // FireEvent.change(select, { target: { value: 'box3d' } })
    // expect(pageTitle.value).toBe('3D Bounding Box')
  })
  test("Proper instructions", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const instructions = getByTestId("instructions") as HTMLInputElement
    const select = getByTestId("label-type") as HTMLSelectElement
    expect(instructions.value).toBeFalsy()
    fireEvent.change(select, { target: { value: "tag" } })
    expect(instructions.value).toBeFalsy()
    fireEvent.change(select, { target: { value: "box2d" } })
    expect(instructions.value).toBe(
      "https://doc.scalabel.ai/instructions/bbox.html"
    )
    fireEvent.change(select, { target: { value: "polygon2d" } })
    expect(instructions.value).toBe(
      "https://doc.scalabel.ai/instructions/segmentation.html"
    )
    fireEvent.change(select, { target: { value: "polyline2d" } })
    expect(instructions.value).toBe(
      "https://doc.scalabel.ai/instructions/segmentation.html"
    )
    fireEvent.change(select, { target: { value: "box3d" } })
    expect(instructions.value).toBeFalsy()
  })
})
describe("Test element hiding", () => {
  test("Hides categories on tagging option", () => {
    const { getByTestId, queryByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const select = getByTestId("label-type") as HTMLSelectElement
    expect(queryByTestId("categories")).not.toBeNull()
    fireEvent.change(select, { target: { value: "tag" } })
    expect((getByTestId("label-type") as HTMLSelectElement).value).toBe("tag")
    expect(queryByTestId("categories")).toBeNull()
  })
  test("Hides and re-shows categories on video option", () => {
    const { getByTestId, queryByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const select = getByTestId("label-type") as HTMLSelectElement
    expect(queryByTestId("categories")).not.toBeNull()
    fireEvent.change(select, { target: { value: "tag" } })
    expect((getByTestId("label-type") as HTMLSelectElement).value).toBe("tag")
    expect(queryByTestId("categories")).toBeNull()
    fireEvent.change(select, { target: { value: "polyline2d" } })
    expect(queryByTestId("categories")).not.toBeNull()
  })
  test("Hides task size on video option", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const select = getByTestId("item-type") as HTMLSelectElement
    const tasksize = getByTestId("tasksize")
    expect(tasksize.className).not.toContain("hidden")
    fireEvent.change(select, { target: { value: "video" } })
    expect((getByTestId("item-type") as HTMLSelectElement).value).toBe("video")
    expect(tasksize.className).toContain("hidden")
  })
  test("Hides and re-shows task size on tagging option", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const select = getByTestId("item-type") as HTMLSelectElement
    const tasksize = getByTestId("tasksize")
    expect(tasksize.className).not.toContain("hidden")
    fireEvent.change(select, { target: { value: "video" } })
    expect((getByTestId("item-type") as HTMLSelectElement).value).toBe("video")
    expect(tasksize.className).toContain("hidden")
    fireEvent.change(select, { target: { value: "image" } })
    expect(tasksize.className).not.toContain("hidden")
  })
  test("Hidden buttons are shown on submission", async () => {
    const { getByTestId, queryByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const form = getByTestId("submit-button")
    expect(queryByTestId("hidden-buttons")).toBeNull()
    fireEvent.click(form)
    expect(xhrMockClass.open).toBeCalled()
  })
})
describe("Test user ability to change fields", () => {
  test("Instruction url cannot be changed by user", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const instructions = getByTestId("instructions") as HTMLInputElement
    const select = getByTestId("label-type") as HTMLSelectElement
    fireEvent.change(select, { target: { value: "box2d" } })
    expect(instructions.value).toBe(
      "https://doc.scalabel.ai/instructions/bbox.html"
    )
    fireEvent.change(instructions, { target: { value: "should not change" } })
    expect(instructions.value).not.toBe("should not change")
  })
  test("Page Title can be changed after it is auto-populated", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const pageTitle = getByTestId("page-title") as HTMLInputElement
    const select = getByTestId("label-type") as HTMLSelectElement
    expect(pageTitle.value).toBeFalsy()
    fireEvent.change(select, { target: { value: "tag" } })
    expect(pageTitle.value).toBe("Image Tagging")
    fireEvent.change(pageTitle, { target: { value: "changed" } })
    expect(pageTitle.value).toBe("changed")
  })
})
describe("Test file uploading", () => {
  test("Uploading file changes filename", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const upload = getByTestId("item_file") as HTMLInputElement
    const filename = getByTestId("item_file_filename") as HTMLInputElement
    const file = new File(["test"], "test.yml")
    Object.defineProperty(upload, "files", {
      value: [file]
    })
    fireEvent.change(upload)
    expect(filename.value).toBe("test.yml")
  })
  test("Cannot change filename", () => {
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <StyledForm />
      </ThemeProvider>
    )
    const upload = getByTestId("item_file") as HTMLInputElement
    const filename = getByTestId("item_file_filename") as HTMLInputElement
    const file = new File(["test"], "test.yml")
    expect(filename.value).toBe("No file chosen")
    Object.defineProperty(upload, "files", {
      value: [file]
    })
    fireEvent.change(upload)
    expect(filename.value).toBe("test.yml")
    fireEvent.change(filename, { target: { value: "should not change" } })
    expect(filename.value).not.toBe("should not change")
  })
})
const StyledForm = withStyles(formStyle)(CreateForm)
