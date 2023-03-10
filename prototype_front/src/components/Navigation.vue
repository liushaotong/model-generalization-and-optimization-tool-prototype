<template>
  <div class="main-container">
    <div class="title-container">
      <h1 class="title">深度学习模型泛化性度量与优化工具原型</h1>
    </div>
    <div class="upload-container">
      <el-upload
        class="upload"
        action="#"
        :show-file-list="false"
        :before-upload="beforeUpload"
      >
        <el-button size="medium">模型上传</el-button>
      </el-upload>
      <el-select v-model="selectedTask" placeholder="请选择图像分类任务" @change=saveTask>
        <el-option label="CIFAR-10" value="cifar-10"></el-option>
        <el-option label="SVHN" value="svhn"></el-option>
      </el-select>
    </div>
    <div class="nav-container">
      <el-menu class="menu" :default-active="$route.path" router >
        <el-menu-item index="/complexity" >复杂度计算</el-menu-item>
        <el-menu-item index="/generalization">泛化性度量</el-menu-item>
        <el-menu-item index="/pruning">剪枝优化</el-menu-item>
      </el-menu>
    </div>
    <router-view></router-view>
  </div>
</template>

<script>
export default {
  data() {
    return {
      selectedTask: null,
      selectKey: 1
    }
  },
  methods: {
    beforeUpload() {
      // 在这里添加模型上传的业务逻辑
    },
    saveTask() {
      switch (this.selectedTask) {
        case 'cifar-10':
          // this.taskName = 'CIFAR-10';
           sessionStorage.setItem("selectedTask", this.selectedTask);
          break;
        case 'svhn':
          // this.taskName = 'SVHN';
           sessionStorage.setItem("selectedTask", this.selectedTask);
          break;
        default:
          // this.taskName = '';
           sessionStorage.setItem("selectedTask", this.selectedTask);
          break;
      }
    }
  }
}
</script>

<style scoped>
.main-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 24px;
  box-sizing: border-box;
}

.title-container {
  width: 100%;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
}

.title {
  font-size: 24px;
  font-weight: bold;
}

.upload-container {
  width: 100%;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
}

.upload {
  margin-right: 24px;
}

.nav-container {
  width: 100%;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
}

.menu {
  border: none;
  background-color: transparent;
  display: flex;
  flex-direction: row;
}
</style>