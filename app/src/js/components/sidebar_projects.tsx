import { Grid, Link } from '@material-ui/core'
import ListItem from '@material-ui/core/ListItem'
import { withStyles } from '@material-ui/core/styles'
import React from 'react'
import { getProjects, toProject } from '../common/service'
import { projectListStyle } from '../styles/create'

interface ClassType {
  /** style for a colored entry */
  coloredListItem: string
}

// props for project list
interface ProjectListProps {
  /** Project List classes */
  classes: ClassType
  /** refresh boolean */
  refresh: boolean
}

interface ProjectListState {
  /** boolean which when changed forces a refresh */
  reloadProjects: boolean
}

/** Project list sidebar component. Re-renders after
 *  submission
 * @param props
 */
class ProjectList extends React.Component
        <ProjectListProps, ProjectListState> {

  /** receive data from backend */
  private projectsToExpress = getProjects()

  public constructor (props: ProjectListProps) {
    super(props)
    this.state = {
      reloadProjects: props.refresh
    }
  }

  /**
   * method to process changes in props, which gets project from backend
   * and then changes state to force a reload of this component
   * @param props
   */
  public componentWillReceiveProps (props: ProjectListProps) {
    const { refresh } = this.props
    if (props.refresh !== refresh) {
      this.projectsToExpress = getProjects()
      this.setState({ reloadProjects: !this.state.reloadProjects })
    }
  }

  /**
   * renders project list
   */
  public render () {
    const { classes } = this.props
    return (
            <div>
              {this.projectsToExpress.map((project, index) => (
                      <ListItem button
                                key={project}
                                alignItems='center'
                                className={!(index % 2) ?
                                        classes.coloredListItem : ''}>
                        < Grid container justify='center'>
                          <Link
                                  component='button'
                                  variant='body2'
                                  color='inherit'
                                  onClick={() => {
                                    toProject(project)
                                  }}
                          >
                            {project}
                          </Link>
                        </Grid>
                      </ListItem>
              ))}
            </div>)
  }
}

export default withStyles(projectListStyle)(ProjectList)
