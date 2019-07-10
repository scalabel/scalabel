import { MuiThemeProvider } from '@material-ui/core/styles'
import { ThemeProvider, withStyles } from '@material-ui/styles'
import { cleanup, fireEvent, render } from '@testing-library/react'
import React from 'react'
import CreateForm from '../components/create_form'
import { formStyle } from '../styles/create'
import { myTheme } from '../styles/theme'

afterEach(cleanup)
describe('Test create page functionality', () => {
  describe('Test different label options', () => {
    test('Proper page titles', () => {
      const { getByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const pageTitle = getByTestId('page-title') as HTMLInputElement
      const select = getByTestId('label-type') as HTMLSelectElement
      expect(pageTitle.value).toBeFalsy()
      fireEvent.change(select, { target: { value: 'tag' } })
      expect(pageTitle.value).toBe('Image Tagging')
      fireEvent.change(select, { target: { value: 'box2d' } })
      expect(pageTitle.value).toBe('2D Bounding Box')
      fireEvent.change(select, { target: { value: 'segmentation' } })
      expect(pageTitle.value).toBe('2D Segmentation')
      fireEvent.change(select, { target: { value: 'lane' } })
      expect(pageTitle.value).toBe('2D Lane')
      fireEvent.change(select, { target: { value: 'box3d' } })
      expect(pageTitle.value).toBe('3D Bounding Box')
    })
    test('Proper instructions', () => {
      const { getByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const instructions = getByTestId('instructions') as HTMLInputElement
      const select = getByTestId('label-type') as HTMLSelectElement
      expect(instructions.value).toBeFalsy()
      fireEvent.change(select, { target: { value: 'tag' } })
      expect(instructions.value).toBeFalsy()
      fireEvent.change(select, { target: { value: 'box2d' } })
      expect(instructions.value).toBe('https://www.scalabel.ai/doc/instructions/bbox.html')
      fireEvent.change(select, { target: { value: 'segmentation' } })
      expect(instructions.value).toBe('https://www.scalabel.ai/doc/instructions/segmentation.html')
      fireEvent.change(select, { target: { value: 'lane' } })
      expect(instructions.value).toBe('https://www.scalabel.ai/doc/instructions/segmentation.html')
      fireEvent.change(select, { target: { value: 'box3d' } })
      expect(instructions.value).toBeFalsy()
    })
  })
  describe('Test element hiding', () => {
    test('Hides categories on tagging option', () => {
      const { getByTestId, queryByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const select = getByTestId('label-type') as HTMLSelectElement
      expect(queryByTestId('categories')).not.toBeNull()
      fireEvent.change(select, { target: { value: 'tag' } })
      expect((getByTestId('label-type') as HTMLSelectElement).value).toBe('tag')
      expect(queryByTestId('categories')).toBeNull()
    })
    test('Hides and re-shows categories on tagging option', () => {
      const { getByTestId, queryByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const select = getByTestId('label-type') as HTMLSelectElement
      expect(queryByTestId('categories')).not.toBeNull()
      fireEvent.change(select, { target: { value: 'tag' } })
      expect((getByTestId('label-type') as HTMLSelectElement).value).toBe('tag')
      expect(queryByTestId('categories')).toBeNull()
      fireEvent.change(select, { target: { value: 'lane' } })
      expect(queryByTestId('categories')).not.toBeNull()
    })
    test('Hidden buttons are shown on submission', () => {
      const { getByTestId, queryByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const form = getByTestId('submit-button')
      expect(queryByTestId('hidden-buttons')).toBeNull()
      fireEvent.click(form)
      expect(queryByTestId('hidden-buttons')).not.toBeNull()
    })
  })
  describe('Test user ability to change fields', () => {
    test('Instruction url cannot be changed by user', () => {
      const { getByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const instructions = getByTestId('instructions') as HTMLInputElement
      const select = getByTestId('label-type') as HTMLSelectElement
      fireEvent.change(select, { target: { value: 'box2d' } })
      expect(instructions.value).toBe('https://www.scalabel.ai/doc/instructions/bbox.html')
      fireEvent.change(instructions, { target: { value: 'should not change' } })
      expect(instructions.value).not.toBe('should not change')
    })
    test('Page Title can be changed after it is auto-populated', () => {
      const { getByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const pageTitle = getByTestId('page-title') as HTMLInputElement
      const select = getByTestId('label-type') as HTMLSelectElement
      expect(pageTitle.value).toBeFalsy()
      fireEvent.change(select, { target: { value: 'tag' } })
      expect(pageTitle.value).toBe('Image Tagging')
      fireEvent.change(pageTitle, { target: { value: 'changed' } })
      expect(pageTitle.value).toBe('changed')
    })
  })
  describe('Test file uploading', () => {
    test('Uploading file changes filename', () => {
      const { getByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const upload = getByTestId('item_file') as HTMLInputElement
      const filename = getByTestId('item_file_filename') as HTMLInputElement
      const file = new File(['test'], 'test.yml')
      Object.defineProperty(upload, 'files', {
        value: [file]
      })
      fireEvent.change(upload)
      expect(filename.value).toBe('test.yml')
    })

    test('Cannot change filename', () => {
      const { getByTestId } = render(
              <MuiThemeProvider theme={myTheme}>
                <ThemeProvider theme={myTheme}>
                  <StyledForm/>
                </ThemeProvider>
              </MuiThemeProvider>)
      const upload = getByTestId('item_file') as HTMLInputElement
      const filename = getByTestId('item_file_filename') as HTMLInputElement
      const file = new File(['test'], 'test.yml')

      expect(filename.value).toBe('No file chosen')
      Object.defineProperty(upload, 'files', {
        value: [file]
      })
      fireEvent.change(upload)
      expect(filename.value).toBe('test.yml')
      fireEvent.change(filename, { target: { value: 'should not change' } })
      expect(filename.value).not.toBe('should not change')
    })
  })
})
const StyledForm = withStyles(formStyle)(CreateForm)
