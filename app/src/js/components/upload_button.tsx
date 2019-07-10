import { Grid, TextField } from '@material-ui/core'
import Button from '@material-ui/core/Button'
import FormControl from '@material-ui/core/FormControl'
import React, { ChangeEvent } from 'react'

interface ClassType {
  /** root class */
  root: string

  /** text field */
  textField: string

  /** filename text */
  filenameText: string

  /** button component root */
  button: string

  /** grid for button and textfield */
  grid: string

}

interface Props {
  /** if the upload is required */
  required: boolean

  /** button Label */
  label: string

  /** form id */
  form_id: string

  /** UploadButton classes */
  classes: ClassType
}

interface State {
  /** filename string */
  filename: string

}

/**
 * Upload file button
 */
export default class UploadButton extends React.Component<Props, State> {

  constructor (props: Props) {
    super(props)
    this.state = { filename: 'No file chosen' }
  }

  /**
   * renders the upload button
   * currently uses native validation
   */
  public render () {
    const { classes } = this.props
    return (
      <div className={classes.root}>
        <label id={this.props.form_id + '_label'} htmlFor={this.props.form_id + '-grid'}> {this.props.label}
        </label>
        <Grid
          container
          id = {this.props.form_id + '-grid'}
          direction='row'
          alignItems='center'
          className={classes.grid}>
          <FormControl>
            <input type='file'
              required={this.props.required}
              id={this.props.form_id} name={this.props.form_id}
                   data-testid = {this.props.form_id}
              style={{
                position: 'absolute',
                width: '1px',
                height: '1px',
                padding: '0',
                margin: '-1px',
                overflow: 'hidden',
                clip: 'rect(0,0,0,0)',
                border: '0'
              }}
              onChange={(event) => this.handleFileChange(event)}
              accept='.yml, .yaml' />
            <label htmlFor={this.props.form_id}>
              <Button
                variant='contained'
                component='span'
                className={classes.button}
                id='test'
              >
                Choose File
              </Button>
            </label>
          </FormControl>
          <TextField value={this.state.filename}
            className={classes.textField}
            InputProps={{
              classes: {
                input: classes.filenameText
              },
              inputProps: {
                'data-testid': this.props.form_id + '_filename'
              }
            }} />
        </Grid>
      </div>
    )
  }

  /**
   * Handle filename change
   * @param event
   */
  private handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      this.setState({ filename: event.target.files[0].name })
    }
  }
}
