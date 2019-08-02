import List from '@material-ui/core/List'
import ListItem from '@material-ui/core/ListItem'
import ListItemText from '@material-ui/core/ListItemText'
import { withStyles } from '@material-ui/core/styles'
import Typography from '@material-ui/core/Typography'
import React, { ReactNode } from 'react'
import { createStyle, formStyle } from '../styles/create'
import CreateForm from './create_form'
import DividedPage from './divided_page'
import ProjectList from './sidebar_projects'

export interface ClassType {
  /** list header (existing projects) */
  listHeader: string
}

export interface Props {
  /** Create classes */
  classes: ClassType
}

export interface State {
  /** boolean to force reload of the sidebar project list */
  reloadProjects: boolean
}

/**
 * Component which display the create page
 * @param {object} props
 * @return component
 */
class Create extends React.Component<Props, State> {
  /**
   * create page header content
   */
  private readonly headerContent: ReactNode
  /**
   * create page sidebar content
   */
  private readonly sidebarContent: ReactNode
  /**
   * create page main content
   */
  private readonly mainContent: ReactNode

  public constructor (props: Props) {
    super(props)
    this.state = {
      reloadProjects: false
    }
    this.headerContent = (<Typography variant='h6' noWrap>
      Create A Project
    </Typography>)
    this.sidebarContent = (
            <List>
              <ListItem>
                <ListItemText primary={'Existing Projects'}
                              className={this.props.classes.listHeader}
                >
                </ListItemText>
              </ListItem>
              <ProjectList
                      refresh={this.state.reloadProjects}/>
            </List>
    )
    this.mainContent = (
            <StyledForm projectReloadCallback={this.projectReloadCallback}/>
    )
  }

  /**
   * renders the create page
   * @return component
   */
  public render () {
    return (
            <DividedPage children={{
              headerContent: this.headerContent,
              sidebarContent: this.sidebarContent,
              mainContent: this.mainContent
            }}/>
    )
  }

  /**
   * callback used to force a state change to reload the project
   * list
   */
  private projectReloadCallback = () => {
    this.setState({ reloadProjects : !this.state.reloadProjects })
  }

}

const StyledForm = withStyles(formStyle)(CreateForm)
/** export Create page */
export default withStyles(createStyle, { withTheme: true })(Create)
