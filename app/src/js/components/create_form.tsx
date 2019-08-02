import { TextField } from '@material-ui/core'
import Button from '@material-ui/core/Button'
import Checkbox from '@material-ui/core/Checkbox'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import FormGroup from '@material-ui/core/FormGroup'
import withStyles from '@material-ui/core/styles/withStyles'
import React, { ChangeEvent } from 'react'
import { sprintf } from 'sprintf-js'
import { attributeStyle, checkboxStyle, uploadStyle } from '../styles/create'
import UploadButton from './upload_button'

interface ClassType {
  /** root class */
  root: string
  /** select class */
  selectEmpty: string
  /** full length text class */
  fullWidthText: string
  /** form row */
  formGroup: string
  /** half length text text */
  halfWidthText: string
  /** form submit button */
  submitButton: string
  /** hidden class, used for categories upload */
  hidden: string
}

interface Props {
  /** styles of the create form */
  classes: ClassType
  /** project list reload callback */
  projectReloadCallback?: () => void
}

interface State {
  /** project name */
  projectName: string
  /** project item type */
  itemType: string
  /** project label type */
  labelType: string
  /** project page title */
  pageTitle: string
  /** current instructions url */
  instructionsUrl: string
  /** dashboard url, change with version update */
  dashboardUrl: string
  /** vendor url, change with version update */
  vendorUrl: string
  /** show advanced options boolean */
  showAdvancedOptions: boolean
  /** demo mode boolean */
  demoMode: boolean
  /** whether the form has been submitted */
  hasSubmitted: boolean
  /** whether to show the categories upload button */
  showCategoriesUpload: boolean
  /** whether to show task size */
  showTaskSize: boolean
}

/**
 * Create Project form
 */
export default class CreateForm extends React.Component<Props, State> {
  constructor (props: Props) {
    super(props)
    this.state = {
      projectName: '',
      itemType: '',
      labelType: '',
      pageTitle: '',
      instructionsUrl: '',
      dashboardUrl: '',
      vendorUrl: '',
      showAdvancedOptions: false,
      demoMode: false,
      hasSubmitted: false,
      showCategoriesUpload: true,
      showTaskSize: true
    }
  }

  /**
   * renders the create form
   */
  public render () {
    const { classes } = this.props
    return (
            <div className={classes.root}>
            <form onSubmit={this.handleSubmit} data-testid='create-form'>
              <FormGroup row={true} className={classes.formGroup}>
                <TextField
                        required
                        name='project_name'
                        value={this.state.projectName}
                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                this.setState(
                                        { projectName: event.target.value })}
                        label='Project Name'
                        inputProps={{
                          'pattern': '[A-Za-z0-9_-]*',
                          'data-testid': 'project-name'
                        }}
                        helperText={'Only letters, numbers, dashes, and ' +
                        'underscores are permitted.'}
                        className={classes.fullWidthText}
                        margin='normal'
                /> </FormGroup>
              <FormGroup row={true} className={classes.formGroup}>
                <TextField
                        value={this.state.itemType}
                        select
                        name='item_type'
                        label='Item Type'
                        onChange={
                          this.handleItemTypeChange
                        }
                        required
                        inputProps={{
                          'data-testid': 'item-type'
                        }}
                        SelectProps={{
                          native: true
                        }}
                        className={classes.selectEmpty}

                >
                  <option/>
                  <option value='image'>Image</option>
                  <option value='video'>Video</option>
                  <option value='pointcloud'>Point Cloud</option>
                  <option value='pointcloudtracking'>
                    Point Cloud Tracking
                  </option>
                </TextField>
                <TextField
                        value={this.state.labelType}
                        select
                        name='label_type'
                        label='Label Type'
                        onChange={this.handleLabelChange}
                        required
                        className={classes.selectEmpty}
                        inputProps={{
                          'data-testid': 'label-type'
                        }}
                        SelectProps={{
                          native: true
                        }}
                >
                  <option/>
                  <option value='tag' data-testid='image-tagging'>
                    Image Tagging
                  </option>
                  <option value='box2d'>2D Bounding Box</option>
                  <option value='segmentation'>Instance
                    Segmentation
                  </option>
                  <option value='lane'>Lane</option>
                  <option value='box3d'>3D Bounding Box</option>
                </TextField>
              </FormGroup>
              <FormGroup row={true} className={classes.formGroup}>
                <TextField
                        value={this.state.pageTitle}
                        name='page_title'
                        label='Page Title'
                        className={classes.fullWidthText}
                        margin='normal'
                        inputProps={{
                          'data-testid': 'page-title'
                        }}
                        onChange={(event: ChangeEvent<HTMLInputElement>) => {
                          this.setState({ pageTitle: event.target.value })
                        }}
                /> </FormGroup>
              <FormGroup row={true} className={classes.formGroup}>
                <StyledUpload required={true}
                              label={'Item List*'}
                              form_id={'item_file'}
                              with_json
                />
                {this.state.showCategoriesUpload ?
                        <StyledUpload required={true}
                                      label={'Categories*'}
                                      form_id={'categories'}/>
                        : null}
                <StyledAttributeUpload required={false}
                                       label={'Attributes'}
                                       form_id={'attributes'}/>
              </FormGroup>
              <FormGroup row={true} className={classes.formGroup}>
                <TextField
                        required={this.state.showTaskSize}
                        type='number'
                        name='task_size'
                        label='Tasksize'
                        className={this.state.showTaskSize ?
                                classes.halfWidthText : classes.hidden}
                        InputProps={{
                          inputProps: {
                            min: 1
                          }
                        }}
                        margin='normal'
                        data-testid= 'tasksize'
                />
              </FormGroup>
              <FormGroup row={true} className={classes.formGroup}>
                <TextField
                        name='instructions'
                        label='Instruction URL'
                        className={classes.fullWidthText}
                        margin='normal'
                        inputProps={{
                          'data-testid': 'instructions'
                        }}
                        value={this.state.instructionsUrl}
                />
              </FormGroup>
              <FormGroup row={true} className={classes.formGroup}>
                <Button variant='contained'
                        component='label'
                        onClick={() => {
                          this.setState({
                            showAdvancedOptions:
                                    !this.state.showAdvancedOptions
                          })
                        }}
                >
                  Show advanced options
                </Button>
              </FormGroup>
              {this.state.showAdvancedOptions ?
                      <FormGroup row={true} className={classes.formGroup}>
                        <FormControlLabel
                                control={
                                  <StyledCheckbox
                                          onChange={() => {
                                            this.setState({
                                              demoMode:
                                                      !this.state.demoMode
                                            })
                                          }}
                                  />
                                }
                                id='demo_mode'
                                name='demo_mode'
                                value={this.state.demoMode}
                                label='Demo Mode'
                                labelPlacement='end'
                        />
                      </FormGroup> : null}
              <FormGroup row={true} className={classes.formGroup}>
                <Button variant='contained'
                        type='submit'
                        data-testid='submit-button'
                        className={classes.submitButton}>
                  Submit
                </Button>
                {this.state.hasSubmitted ?
                        <div id='hidden-buttons' data-testid='hidden-buttons'>
                          <Button variant='contained'
                                  href={this.state.dashboardUrl}
                                  id='go_to_dashboard'
                                  data-testid='dashboard-button'
                                  className={classes.submitButton}
                          >
                            Go to Dashboard
                          </Button>
                          <Button variant='contained'
                                  href={this.state.vendorUrl}
                                  id='go_to_vendor_dashboard'
                                  data-testid='vendor-button'
                                  className={classes.submitButton}>
                            Go to Vendor
                            Dashboard</Button>
                        </div>
                        : null}
              </FormGroup>
            </form>
            </div>
    )
  }

  /**
   * Handles submission event
   * @param event
   */
  private handleSubmit = (event: ChangeEvent<HTMLFormElement>) => {
    event.preventDefault()
    const that = this
    const x = new XMLHttpRequest()
    x.onreadystatechange = () => {
      if (x.readyState === 4) {
        if (x.response) {
          alert(x.response)
        } else {
          that.setState((prevState: State) => ({
            projectName: prevState.projectName.replace(
                    new RegExp(' ', 'g'), '_')
          }))
          that.setState((prevState: State) => ({
            dashboardUrl: './dashboard?project_name='
                    + prevState.projectName
          }))
          that.setState((prevState: State) => ({
            vendorUrl: './vendor?project_name='
                    + prevState.projectName
          }))
          if (that.props.projectReloadCallback) {
            that.props.projectReloadCallback()
          }
          return
        }
      }
    }
    // CHANGE AFTER VERSION UPDATE
    const version = 'v1'
    x.open('POST', sprintf('./postProject?v=%s', version))
    const formData = new FormData(event.target)
    x.send(formData)
    if (!this.state.hasSubmitted) {
      this.setState({ hasSubmitted: true })
    }
  }
  /**
   * handles instruction url
   * @param itemType {string}
   */
  private handleInstructions = (itemType: string) => {
    let labelName = ''
    let instructions = ''
    if (itemType === 'tag') {
      labelName = 'Image Tagging'
      this.setState({ showCategoriesUpload: false })
    } else if (itemType === 'box2d') {
      labelName = '2D Bounding Box'
      instructions = 'https://www.scalabel.ai/doc/instructions/bbox.html'
      this.setState({ showCategoriesUpload: true })
    } else if (itemType === 'segmentation') {
      labelName = '2D Segmentation'
      instructions = 'https://www.scalabel.ai/doc/instructions/' +
              'segmentation.html'
      this.setState({ showCategoriesUpload: true })
    } else if (itemType === 'lane') {
      labelName = '2D Lane'
      instructions = 'https://www.scalabel.ai/doc/instructions/' +
              'segmentation.html'
      this.setState({ showCategoriesUpload: true })
    } else if (itemType === 'box3d') {
      labelName = '3D Bounding Box'
      this.setState({ showCategoriesUpload: true })
    }
    this.setState({ pageTitle: labelName })
    this.setState({ instructionsUrl: instructions })
  }
  /**
   * handles label changing
   * @param event
   */
  private handleLabelChange = (event: ChangeEvent<HTMLSelectElement>) => {
    this.handleInstructions(event.target.value)
    this.setState({ labelType: event.target.value })
  }
  /**
   * handles item type changing
   * @param event
   */
  private handleItemTypeChange = (event: ChangeEvent<HTMLSelectElement>) => {
    this.setState({ itemType: event.target.value })
    if (event.target.value === 'video') {
      this.setState({ showTaskSize: false })
    } else {
      this.setState({ showTaskSize: true })
    }
  }
}
const StyledCheckbox = withStyles(checkboxStyle)(Checkbox)
const StyledUpload = withStyles(uploadStyle)(UploadButton)
/* Need to have separate class for attribute upload in the case where
   the categories upload is hidden */
const StyledAttributeUpload = withStyles(attributeStyle)(StyledUpload)
