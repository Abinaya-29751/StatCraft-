# 🚀 Vercel Deployment Guide

## Prerequisites

### 1. Get Google Gemini API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key (starts with "AIza...")

### 2. Push to GitHub
Make sure your code is pushed to GitHub repository:
```bash
git add .
git commit -m "Ready for Vercel deployment"
git push origin main
```

## 🚀 Vercel Deployment Steps

### Step 1: Create Vercel Account
1. Go to: https://vercel.com
2. Click "Sign Up"
3. Sign up with your GitHub account
4. Authorize Vercel to access your repositories

### Step 2: Import Project
1. Click "New Project"
2. Select "Import Git Repository"
3. Choose your repository: `MULTIMETA/dataset-analyzer`
4. Click "Import"

### Step 3: Configure Project

**Framework Preset:**
- **Framework**: `Other` (since it's FastAPI)
- **Root Directory**: `./` (root of your project)
- **Build Command**: `pip install -r requirements.txt`
- **Output Directory**: Leave empty (not needed for Python)

**Environment Variables:**
- **GOOGLE_API_KEY**: Your actual Google API key (set in Vercel dashboard)
- **HOST**: `0.0.0.0`
- **PORT**: `8000`

### Step 4: Set Environment Variables
**IMPORTANT**: Before deploying, set your environment variables:

1. **In Vercel Dashboard**:
   - Go to your project settings
   - Click "Environment Variables"
   - Add new variable:
     - **Name**: `GOOGLE_API_KEY`
     - **Value**: Your actual Google API key
     - **Environment**: Production, Preview, Development (select all)
   - Click "Save"

2. **Optional variables**:
   - **HOST**: `0.0.0.0`
   - **PORT**: `8000`

### Step 5: Deploy
1. Click "Deploy"
2. Vercel will automatically:
   - Install dependencies from requirements.txt
   - Build your FastAPI application
   - Deploy it globally
3. Wait for deployment to complete (usually 1-3 minutes)

### Step 5: Test Your Deployment
1. Once deployed, you'll get a URL like: `https://dataset-analyzer-xyz.vercel.app`
2. Visit the URL to test your app
3. Upload a sample CSV file to verify everything works

## 🔧 Vercel-Specific Configuration

### Serverless Functions
- **Runtime**: Python 3.9
- **Memory**: 1024MB (default)
- **Timeout**: 10 seconds (default)
- **Region**: Global (automatic)

### File Upload Limits
- **Free Tier**: 4.5MB max file size
- **Pro Plan**: 50MB max file size
- **Enterprise**: Custom limits

### Environment Variables
Set these in Vercel dashboard:
- **GOOGLE_API_KEY**: Your Google Gemini API key
- **HOST**: `0.0.0.0`
- **PORT**: `8000`

## 🆘 Troubleshooting

### Common Issues:

1. **Build Fails**: 
   - Check requirements.txt
   - Ensure all dependencies are listed
   - Check Python version compatibility

2. **Function Timeout**:
   - Large datasets may timeout
   - Consider upgrading to Pro plan
   - Optimize your analysis code

3. **API Key Error**:
   - Verify GOOGLE_API_KEY is set correctly
   - Check environment variables in Vercel dashboard

4. **File Upload Issues**:
   - Check file size limits
   - Ensure proper multipart handling

### Checking Logs:
1. Go to your project dashboard
2. Click "Functions" tab
3. Click on your function
4. Check "Logs" for error messages

### Performance Optimization:
- Use smaller datasets for testing
- Consider upgrading to Pro plan for production
- Monitor function execution time
- Use Vercel Analytics for insights

## 📊 Vercel Dashboard Features

### Monitoring:
- **Analytics**: Page views, performance metrics
- **Functions**: Execution time, memory usage
- **Logs**: Real-time function logs
- **Deployments**: Build and deployment history

### Management:
- **Environment Variables**: Secure storage
- **Custom Domains**: Add your own domain
- **SSL**: Automatic HTTPS
- **Edge Functions**: Global distribution

## 🎯 Success Checklist

- [ ] Google API key obtained
- [ ] Code pushed to GitHub
- [ ] Vercel account created
- [ ] Project imported from GitHub
- [ ] Environment variables set
- [ ] Deployment successful
- [ ] App accessible via URL
- [ ] File upload tested
- [ ] Analysis functionality verified

## 💡 Pro Tips

1. **Start with Free Tier**: Test everything first
2. **Monitor Function Logs**: Check for any issues
3. **Test with Small Files**: Verify functionality
4. **Use Vercel Analytics**: Monitor performance
5. **Set Up Custom Domain**: For production use

## 🚀 Next Steps After Deployment

1. **Test All Features**: Upload, analyze, visualize
2. **Share Your App**: Get the public URL
3. **Monitor Performance**: Check Vercel dashboard
4. **Scale if Needed**: Upgrade to Pro plan for production

## 🔄 Auto-Deploy

Vercel automatically deploys when you:
- Push to main branch
- Create pull requests
- Merge pull requests

Your Dataset Analyzer will be live and accessible globally! 🌍

## 📈 Vercel Plans

### Free Tier:
- ✅ 100GB bandwidth/month
- ✅ 4.5MB file upload limit
- ✅ Serverless functions
- ✅ Automatic HTTPS

### Pro Plan ($20/month):
- ✅ 1TB bandwidth/month
- ✅ 50MB file upload limit
- ✅ Advanced analytics
- ✅ Priority support

### Enterprise:
- ✅ Custom limits
- ✅ Dedicated support
- ✅ Advanced security
