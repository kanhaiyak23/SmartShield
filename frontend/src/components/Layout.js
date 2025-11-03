import Head from 'next/head';

export default function Layout({ children }) {
  return (
    <>
      <Head>
        <title>SmartShield - Campus Intrusion Detection</title>
        <meta name="description" content="AI-Powered Campus Intrusion Detection System" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <div className="flex-shrink-0 flex items-center">
                  <span className="text-2xl font-bold text-blue-600">🛡️ SmartShield</span>
                </div>
              </div>
              <div className="flex items-center space-x-6">
                <a href="#" className="text-gray-700 hover:text-blue-600 transition">
                  Dashboard
                </a>
                <a href="#" className="text-gray-700 hover:text-blue-600 transition">
                  Alerts
                </a>
                <a href="#" className="text-gray-700 hover:text-blue-600 transition">
                  Analytics
                </a>
              </div>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </div>
    </>
  );
}

