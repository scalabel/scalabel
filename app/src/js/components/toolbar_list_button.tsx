import { ListItemText } from '@material-ui/core'
import List from '@material-ui/core/List'
import ListItem from '@material-ui/core/ListItem'
import { withStyles } from '@material-ui/core/styles'
import ToggleButton from '@material-ui/lab/ToggleButton'
import ToggleButtonGroup from '@material-ui/lab/ToggleButtonGroup'
import _ from 'lodash'
import React from 'react'
import { listButtonStyle, toggleButtonStyle } from '../styles/label'

interface ClassType {
  /** root class */
  root: string
  /** element containg toggle class */
  toggleContainer: string
  /** button group class */
  buttonGroup: string
  /** primary text class */
  primary: string
  /** toggle button class */
  toggleButton: string
}

interface Props {
  /** handles attribute toggling */
  handleAttributeToggle: (
    toggleName: string,
    alignment: string
  ) => void
  /** styles of ToggleButtons */
  classes: ClassType
  /** name of ToggleButtons */
  name: string
  /** values of ToggleButtons */
  values: string[]
  /** initial alignment index */
  initialAlignmentIndex: number
}

/**
 * This is ToggleButtons component that displays
 * the everything post in the dashboard.
 * @param {object} props
 */
class ToggleButtons extends React.Component<Props> {
  /** state of ToggleButtons */
  public state = {
    alignment: this.props.values[this.props.initialAlignmentIndex]
  }
  /** handleAlignment of ToggleButtons that align buttons */
  public handleAlignment = (
    _event: React.MouseEvent<HTMLElement>,
    alignment: string
  ) => {
    this.props.handleAttributeToggle(
      this.props.name,
      alignment
    )
    this.setState({ alignment })
  }

  /** render function of ToggleButtons */
  public render () {
    const { alignment } = this.state
    const { name, classes, values } = this.props
    const ToggleBtn = withStyles(toggleButtonStyle)(ToggleButton)
    return (
      <List style={{ width: '100%', padding: '0px' }}>
        <ListItemText
          style={{ textAlign: 'center', width: '100%' }}
          classes={{ primary: classes.primary }}
          primary={name}
        />
        <ListItem style={{ width: '100%' }} dense={true}>
          <div
            className={classes.toggleContainer}
            style={{
              marginRight: 'auto',
              marginLeft: 'auto'
            }}
          >
            <ToggleButtonGroup
              className={classes.buttonGroup}
              value={alignment}
              exclusive
              onChange={this.handleAlignment}
            >
              {values.map((element: string) => (
                <ToggleBtn className={classes.toggleButton} value={element}
                key={element}
                data-testid={'toggle-button-' + element}>
                  {' '}
                  {element}{' '}
                </ToggleBtn>
              ))}
            </ToggleButtonGroup>
          </div>
        </ListItem>
      </List>
    )
  }

  /**
   * updates props when re-rendered
   * @param nextProps
   */
  public componentWillReceiveProps (nextProps: Props) {
    this.setState({alignment: nextProps.values[nextProps.initialAlignmentIndex]
    })
  }
}

export const ListButton = withStyles(listButtonStyle)(ToggleButtons)
