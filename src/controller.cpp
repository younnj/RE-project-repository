#include "controller.h"

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>

cRoboticsController::cRoboticsController(const std::string& urdf_path, 
                                         const std::string& manipulator_control_mode, 
                                         const double& dt)
: manipulator_control_mode_(manipulator_control_mode),
  dt_(dt)
{
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);

    // Initialize joint space state
    q_ = VectorXd::Zero(model_.nq);
    qdot_ = VectorXd::Zero(model_.nq);
    tau_ = VectorXd::Zero(model_.nq);
    q_desired_ = VectorXd::Zero(model_.nq);
    tau_desired_ = VectorXd::Zero(model_.nq);
    q_init_ = VectorXd::Zero(model_.nq);
    qdot_init_ = VectorXd::Zero(model_.nq);

    // Initialize task space state
    x_ = Matrix4d::Identity();
    xdot_ = VectorXd::Zero(6);
    J_ = MatrixXd::Zero(6, model_.nv);
    x2_ = Matrix4d::Identity();
    x2dot_ = VectorXd::Zero(6);
    J2_ = MatrixXd::Zero(6, model_.nv);

    // Initialize joint space dynamics
    M_ = MatrixXd::Zero(model_.nq, model_.nv);
    g_ = VectorXd::Zero(model_.nq);       
    c_ = VectorXd::Zero(model_.nq);       

    logging_file_.open("logging.txt");
}

void cRoboticsController::keyMapping(const int &key)
{
    switch (key)
    {
    // Implement with user input
    case '1':
       setMode(joint_ctrl_init);
        break;
    case '2':
       setMode(joint_ctrl_home);
        break;
    
    // --------------------------------------------------------------------------------------
    // TODO 3: Add your keyboard mapping here (using the control modes from TODO 1)
    // Example:
    case '3':
       setMode(hw2_1);
        break;
    case '4':
       setMode(torque_ctrl_dynamic);
        break;
    // --------------------------------------------------------------------------------------
    
    default:
        break;
    }
}

void cRoboticsController::compute(const double& play_time)
{
    play_time_ = play_time;
    if(is_mode_changed_)
    {
        is_mode_changed_ = false;
        control_start_time_ = play_time_;
        q_init_ = q_;
        qdot_init_ = qdot_;
        x_init_ = x_;
        x2_init_ = x2_;
    }

    switch (control_mode_)
    {
        case joint_ctrl_init:
        {
            Vector7d target_position;
            target_position << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_PI / 4.;
            if(manipulator_control_mode_ == "position") moveJointPosition(target_position, 2.0);
            else moveJointPositionTorque(target_position, 2.0);
            break;
        }
        case joint_ctrl_home:
        {
            Vector7d target_position;
            target_position << 0.0, 0.0, 0.0, -M_PI/2., 0.0, M_PI/2., M_PI / 4.;
            if(manipulator_control_mode_ == "position") moveJointPosition(target_position, 2.0);
            else moveJointPositionTorque(target_position, 2.0);
            break;
        }
        // ----------------------------------------------------------------------------
        // TODO 4: Add your control logic here (using the control functions from TODO 5)
        // Example:
        case hw2_1:
        {
            HW2_1();
        }
        case torque_ctrl_dynamic:
        {
            torqueCtrlDynamic();
        }
        // ----------------------------------------------------------------------------
        
        default:
            if(manipulator_control_mode_ == "torque") tau_desired_ = g_;
            break;
    }
}

// =============================================================================
// =============================== User functions ==============================
// =============================================================================
void cRoboticsController::moveJointPosition(const VectorXd& target_position, double duration)
{
	Vector7d zero_vector;
	zero_vector.setZero();
	q_desired_ = DyrosMath::cubicVector<7>(play_time_,
		                                   control_start_time_,
		                                   control_start_time_ + duration, 
                                           q_init_, 
                                           target_position, 
                                           zero_vector, 
                                           zero_vector);
}

void cRoboticsController::moveJointPositionTorque(const VectorXd& target_position, double duration)
{
	Matrix7d kp, kv;
	Vector7d q_cubic, qd_cubic;
	
	kp = Matrix7d::Identity() * 500.0;
	kv = Matrix7d::Identity() * 20.0;

	for (int i = 0; i < 7; i++)
	{
		qd_cubic(i) = DyrosMath::cubicDot(play_time_, 
                                          control_start_time_,
			                              control_start_time_ + duration, q_init_(i), 
                                          target_position(i), 
                                          0, 
                                          0);
		q_cubic(i) = DyrosMath::cubic(play_time_,
                                      control_start_time_,
			                          control_start_time_ + duration, 
                                      q_init_(i), 
                                      target_position(i), 
                                      0, 
                                      0);
	}


	tau_desired_ = M_ * (kp*(q_cubic - q_) + kv*(qd_cubic - qdot_)) + g_;
}

// -------------------------------------
// TODO 5: Add your control functions here
// Example:
void cRoboticsController::HW2_1()
{
    // logging_file_ << ... << std::endl;
    // q_desired_ = 
}

void cRoboticsController::torqueCtrlDynamic()
{
    // logging_file_ << ... << std::endl;
    // torque_desired_ = 
}
// -------------------------------------
// =============================================================================

void cRoboticsController::setMode(const CTRL_MODE& control_mode)
{
    is_mode_changed_ = true;
    control_mode_ = control_mode;
    std::cout << "Control mode change: " << control_mode_ << std::endl;
}

void cRoboticsController::printState()
{
    // TODO 6: Extend or modify this for debugging your controller
    std::cout << "\n\n------------------------------------------------------------------" << std::endl;
    std::cout << "time     : " << std::fixed << std::setprecision(3) << play_time_ << std::endl;
    std::cout << "q now    :\t";
    std::cout << std::fixed << std::setprecision(3) << q_.transpose() << std::endl;
    std::cout << "q desired:\t";
    std::cout << std::fixed << std::setprecision(3) << q_desired_.transpose() << std::endl;
    std::cout << "x        :\n";
    std::cout << std::fixed << std::setprecision(3) << x_ << std::endl;
    std::cout << "x dot    :\t";
    std::cout << std::fixed << std::setprecision(3) << xdot_.transpose() << std::endl;
    std::cout << "J        :\n";
    std::cout << std::fixed << std::setprecision(3) << J_ << std::endl;
}

Matrix4d cRoboticsController::getLinkPose(const VectorXd& q, const std::string& link_name)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEPose Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return Matrix4d::Identity();
    }
    pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
    if (link_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << link_name << " not found in URDF." << std::endl;
        return Matrix4d::Identity();
    }

    pinocchio::Data data_tmp(model_);
    pinocchio::framesForwardKinematics(model_, data_tmp, q);
    return data_tmp.oMf[link_index].toHomogeneousMatrix();
}

MatrixXd cRoboticsController::getLinkJac(const VectorXd& q, const std::string& link_name)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEJac Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }
    pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
    if (link_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << link_name << " not found in URDF." << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }

    MatrixXd J;
    J.setZero(6, model_.nv);
    pinocchio::Data data_tmp(model_);
    pinocchio::computeJointJacobians(model_, data_tmp, q);
    pinocchio::getFrameJacobian(model_, data_tmp, link_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);

    return J;
}

Matrix4d cRoboticsController::getEEPose(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEPose Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return Matrix4d::Identity();
    }
    pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
    if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << ee_name_ << " not found in URDF." << std::endl;
        return Matrix4d::Identity();
    }

    pinocchio::Data data_tmp(model_);
    pinocchio::framesForwardKinematics(model_, data_tmp, q);
    return data_tmp.oMf[ee_index].toHomogeneousMatrix();
}

MatrixXd cRoboticsController::getEEJac(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEJac Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }
    pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
    if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << ee_name_ << " not found in URDF." << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }

    MatrixXd J;
    J.setZero(6, model_.nv);
    pinocchio::Data data_tmp(model_);
    pinocchio::computeJointJacobians(model_, data_tmp, q);
    pinocchio::getFrameJacobian(model_, data_tmp, ee_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);

    return J;
}

MatrixXd cRoboticsController::getMassMatrix(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getMassMatrix Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return MatrixXd::Zero(model_.nq, model_.nq);
    }
    pinocchio::Data data_tmp(model_);
    pinocchio::crba(model_, data_tmp, q);

    return data_tmp.M.selfadjointView<Upper>();  // Only upper triangular part of M_ is computed by pinocchio::crba
}

VectorXd cRoboticsController::getGravityVector(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getGravityVector Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return VectorXd::Zero(model_.nq);
    }
    pinocchio::Data data_tmp(model_);
    pinocchio::computeGeneralizedGravity(model_, data_tmp, q);

    return data_tmp.g;
}

bool cRoboticsController::updateModel(const VectorXd& q, const VectorXd& qdot, const VectorXd& tau)
{
    q_ = q;
    qdot_ = qdot;
    tau_ = tau;
    if(!updateKinematics(q_, qdot_)) return false;
    if(!updateDynamics(q_, qdot_)) return false;

    return true;
}

bool cRoboticsController::updateKinematics(const VectorXd& q, const VectorXd& qdot)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "updateKinematics Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return false;
    }
    if(qdot.size() != model_.nv)
    {
        std::cerr << "updateKinematics Error: size of qdot " << qdot.size() << " is not equal to model.nv size: " << model_.nv << std::endl;
        return false;
    }
    pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
    if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << ee_name_ << " not found in URDF." << std::endl;
        return false;
    }

    pinocchio::FrameIndex link4_index = model_.getFrameId(link_names_[4]);
    if (link4_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << link_names_[4] << " not found in URDF." << std::endl;
        return false;
    }

    pinocchio::computeJointJacobians(model_, data_, q);

    pinocchio::getFrameJacobian(model_, data_, ee_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_);
    x_ = data_.oMf[ee_index].toHomogeneousMatrix();
    xdot_ = J_ * qdot;

    pinocchio::getFrameJacobian(model_, data_, link4_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J2_);
    x2_ = data_.oMf[link4_index].toHomogeneousMatrix();
    x2dot_ = J2_ * qdot;

    return true;
}

bool cRoboticsController::updateDynamics(const VectorXd& q, const VectorXd& qdot)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "updateDynamics Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return false;
    }
    if(qdot.size() != model_.nv)
    {
        std::cerr << "updateDynamics Error: size of qdot " << qdot.size() << " is not equal to model.nv size: " << model_.nv << std::endl;
        return false;
    }
    pinocchio::crba(model_, data_, q);
    pinocchio::computeGeneralizedGravity(model_, data_, q);
    pinocchio::computeCoriolisMatrix(model_, data_, q, qdot);

    // update joint space dynamics
    M_ = data_.M;
    M_ = M_.selfadjointView<Upper>();  // Only upper triangular part of M_ is computed by pinocchio::crba
    g_ = data_.g;
    c_ = data_.C * qdot_;

    return true;
}

VectorXd cRoboticsController::getCtrlInput()
{
    if(manipulator_control_mode_ == "position")
    {
        return q_desired_;
    }
    else
    {
        return tau_desired_;
    }
}