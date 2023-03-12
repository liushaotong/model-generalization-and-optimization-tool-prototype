<template>
  <div class="page-container">
    <el-row>
      <el-col :span="24">
        <el-card class="metrics-card">
          <div slot="header">
            <span class="card-title">剪枝优化结果</span>
          </div>
          <div>
            <el-button type="primary" :loading="loading" @click="loadData">开始剪枝</el-button>
            <el-button type="primary" icon="el-icon-download" :disabled="!downloadable" @click="downloadModel">下载模型</el-button>
          </div>
          <div class="metric-item" v-for="(value, key) in metrics" :key="key">
            <div class="metric-name">{{ key }}</div>
            <div class="metric-value">{{ value }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>
<script>
export default {
  data() {
    return {
      metrics: {},
      loading: false,
      downloadable: false,
      downloadUrl: "model.tar" // 修改为你想要下载的模型文件路径
    };
  },
  methods: {
    loadData() {
      this.loading = true;
      this.metrics = {};
      let dict = {};
      const selectedTask = sessionStorage.getItem('selectedTask');
      dict['selectedTask'] = selectedTask;
      this.$http.post('http://localhost:5000/pruning', dict)
        .then(response => {
          this.metrics = response.data;
          // 保存数据到 session
          sessionStorage.setItem('pruningMetrics', JSON.stringify(this.metrics));
          this.downloadable = true;
        })
        .catch(error => {
          console.log(error);
        })
        .finally(() => {
          this.loading = false; // 请求结束后将loading设置为false
        });
    },
    loadFromSession() {
      // 从 session 获取数据
      const metrics = sessionStorage.getItem('pruningMetrics');
      if (metrics) {
        this.metrics = JSON.parse(metrics);
        this.downloadable = true;
      }
    },
    downloadModel() {
      // window.open(this.downloadUrl);
  //     const link = document.createElement('a');
  // link.href = this.downloadUrl;
  // link.download = this.downloadUrl.substr(this.downloadUrl.lastIndexOf('/') + 1);
  // link.setAttribute('target', '_blank');
  // link.setAttribute('rel', 'noopener noreferrer');
  // link.setAttribute('type', 'application/octet-stream');
  // document.body.appendChild(link);
  // link.click();
  // document.body.removeChild(link);
  this.$http({
    url: 'http://localhost:5000/download',
    method: 'GET',
    responseType: 'blob',
  }).then(response => {
    const url = URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.download = this.downloadUrl;
    link.click();
  }).catch(error => {
    console.log(error);
  });
    }
  },
  created() {
    // 在组件创建时从 session 中加载数据
    this.loadFromSession();
  }
};
</script>
<style scoped>
.page-container {
  padding: 20px;
}

.metrics-card {
  margin-bottom: 20px;
}

.card-title {
  font-size: 18px;
  font-weight: bold;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #ebeef5;
  padding: 10px 0;
}

.metric-name {
  flex: 1;
  font-size: 16px;
  font-weight: bold;
}

.metric-value {
  font-size: 16px;
  color: #606266;
}
</style>




