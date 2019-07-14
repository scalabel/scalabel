import { Grid, Link } from '@material-ui/core'
import ListItem from '@material-ui/core/ListItem'
import React from 'react'
import { getProjects, toProject } from '../common/service'
import { Props, State } from './create_project'

// props for project list
interface ProjectListProps extends Props {
  /** refresh boolean */
  refresh: boolean
}

/** Project list sidebar component. Re-renders after
 *  submission
 * @param props
 */
export default class ProjectList extends React.Component
        <ProjectListProps, State> {

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
