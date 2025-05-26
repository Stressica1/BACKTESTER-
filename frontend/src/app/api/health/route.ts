import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function GET(request: NextRequest) {
  return NextResponse.json({ 
    status: 'ok',
    timestamp: new Date().toISOString(),
    message: 'Trading Platform Frontend API is running'
  });
}

export async function POST(request: NextRequest) {
  return NextResponse.json({ 
    message: 'Method not allowed' 
  }, { status: 405 });
}
